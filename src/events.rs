use std::num::ParseFloatError;
use std::path::PathBuf;

use glium::draw_parameters::DrawParameters;
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;
use glium::{Surface, Blend, Display};
use glium::framebuffer::SimpleFrameBuffer;
use glium::texture::{SrgbTexture2d, RawImage2d, SrgbFormat, MipmapsOption};

use image::{ImageBuffer, Rgba};

use webweaver::Layout;

use nalgebra::{Eye, rotate, rotation_between};

use daggy::{Walker, NodeIndex};
use daggy::petgraph::EdgeDirection;

use texturegen::palette::pixel::Srgb;
use texturegen::{Col, Generator, port};
use texturegen::process::{Process, Constant, Stripes, BlendType, EdgeDetect, Select, EdgeDetectType, VoronoiNoise, Noise, Setting, SettingMut, Invert};
use texturegen::process::Blend as BlendProcess;

use {SimContext, Selection, Node, Vect, Mat, input_pos, output_pos};
use graphics::RenderContext;
use State::*;
use math::*;

pub fn handle(display: &Display, rctx: &RenderContext, gen: &mut Generator<Node>, ctx: &mut SimContext) {
    use glium::glutin::Event::*;
    use glium::glutin::ElementState::*;
    use glium::glutin::MouseButton as Mouse;
    use glium::glutin::VirtualKeyCode as Key;
    use glium::glutin::MouseScrollDelta;
    for event in display.poll_events() {
        match event {
            Closed => ctx.running = false,
            ReceivedCharacter(c) => {
                if let Some(Writing) = ctx.state {
                    if !c.is_whitespace() && !c.is_control() {
                        if ctx.caret == ctx.text.len() {
                            ctx.text.push(c);
                        } else {
                            ctx.text.insert(ctx.caret, c);
                        }
                        if ctx.caret < ctx.text.len() {
                            ctx.caret += 1;
                        }
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Right)) => {
                if let Some(Writing) = ctx.state {
                    if ctx.caret < ctx.text.len() {
                        ctx.caret += 1;
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Left)) => {
                if let Some(Writing) = ctx.state {
                    if ctx.caret > 0 {
                        ctx.caret -= 1;
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Back)) => {
                if let Some(Writing) = ctx.state {
                    if ctx.caret > 0 {
                        ctx.text.remove(ctx.caret - 1);
                        ctx.caret -= 1;
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Delete)) => {
                if let Some(Writing) = ctx.state {
                    if ctx.caret < ctx.text.len() {
                        ctx.text.remove(ctx.caret);
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Escape)) => {
                if let Some(Writing) = ctx.state {
                    ctx.text.clear();
                    ctx.state = None;
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Return)) => {
                use texturegen::process::SettingMut::*;
                if let Some(Writing) = ctx.state {
                    if let Some(selected) = ctx.selected {
                        if let Selection::Setting(n, i) = selected {
                            let mut n = gen.get_process_mut(n).unwrap();
                            let setting = n.settings()[i];
                            match n.setting_mut(setting) {
                                Text(t) => {
                                    *t = ctx.text.to_lowercase();
                                },
                                Integer(i) => {
                                    if let Ok(ii) = ctx.text.parse() {
                                        *i = ii;
                                    }
                                },
                                Float(f) => {
                                    if let Ok(ff) = ctx.text.parse() {
                                        *f = ff;
                                    }
                                },
                                Color(c) => {
                                    if let Ok(col) = decode_color(&ctx.text) {
                                        *c = col.into();
                                    }
                                },
                                _ => {}
                            }
                        }
                        ctx.text.clear();
                        ctx.state = None;
                        if let Some(n) = selected.node() {
                            ctx.selected = Some(Selection::Node(n));
                        }
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Tab)) => {
                let layout = Layout::layout(gen.graph(), 0.25, |from, to| {
                    let (edge, dir) = gen.graph().find_edge_undirected(from, to).unwrap();
                    let edge = gen.graph().edge_weight(edge).unwrap();
                    if let EdgeDirection::Outgoing = dir {
                        output_pos(&gen, port(from, edge.source), ctx.thingy_size)
                    } else {
                        input_pos(&gen, port(to, edge.target), ctx.thingy_size)
                    }
                });
                let mut rot = None;
                for i in 0..gen.graph().node_count() {
                    let i = NodeIndex::new(i);
                    let mut pos: Vect = *layout.get(i).unwrap();
                    if rot.is_none() {
                        rot = Some(rotation_between(&pos, &Vect::new(1., 0.)));
                    }
                    if let Some(ref rot) = rot {
                        pos = rotate(rot, &pos);
                    }
                    gen.get_data_mut(i).unwrap().pos = pos;
                }
            },
            KeyboardInput(Pressed, _, Some(Key::C)) => {
                if let None = ctx.state {
                    if let Some(Selection::Node(selected)) = ctx.selected {
                        // TODO: This is ugly... Would non-lexical borrowing fix it?
                        let process = if let Some((process, _)) = gen.get(selected) {
                            Some(process.clone())
                        } else {
                            None
                        };
                        if let Some(process) = process {
                            ctx.selected = Some(Selection::Node(gen.add(process, Node::new(ctx.mouse_pos))));
                        }
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::E)) => {
                if let None = ctx.state {
                    if let Some(Selection::Node(selected)) = ctx.selected {
                        let draw_params = DrawParameters {
                            blend: Blend {
                                color: Addition {
                                    source: SourceAlpha,
                                    destination: OneMinusSourceAlpha,
                                },
                                alpha: Addition {
                                    source: SourceAlpha,
                                    destination: OneMinusSourceAlpha,
                                },
                                constant_value: (0f32, 0f32, 0f32, 1f32),
                            },
                            smooth: None,
                            ..Default::default()
                        };
                        let (width, height) = (1024, 1024); // TODO: Way to specify these.
                        let texture = SrgbTexture2d::empty_with_format(display, SrgbFormat::U8U8U8U8, MipmapsOption::NoMipmap, width, height).unwrap();
                        let mut target = SimpleFrameBuffer::new(display, &texture).unwrap();
                        let program = gen.get(selected).unwrap().1.shader.borrow();
                        let program = program.as_ref().expect("Node didn't have shader.");
                        let mut matrix = Mat::new_identity(4);
                        matrix.m14 = -1.;
                        matrix.m24 = -1.;
                        matrix.m11 = 2.;
                        matrix.m22 = 2.;
                        let uniforms = uniform! {
                            matrix: *matrix.as_ref(),
                        };
                        let model = rctx.models.get("node").unwrap();
                        target.draw(&model.vertices, &model.indices, &program, &uniforms, &draw_params).expect("Drawing node failed.");
                        let texture = texture.read::<RawImage2d<_>>();
                        let buffer = ImageBuffer::from_fn(width, height, |x, y| {
                            let i = ((height - y - 1) * width + x) as usize * 4;
                            let col = &texture.data;
                            Rgba {
                                data: [col[i + 0], col[i + 1], col[i + 2], col[i + 3]],
                            }
                        });
                        let mut p = PathBuf::new();
                        p.push("textures");
                        let _ = ::std::fs::create_dir(p.clone());
                        p.push("texture");
                        buffer.save(p.with_extension("png")).ok().unwrap();
                    }
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key1)) => {
                if let None = ctx.state {
                    let node = gen.add(Constant::new(Col::new(1., 1., 1.)), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key2)) => {
                if let None = ctx.state {
                    let node = gen.add(BlendProcess::new(BlendType::Screen, BlendType::Normal), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key3)) => {
                if let None = ctx.state {
                    let node = gen.add(Stripes::new(4, 1, Col::new(1., 1., 1.), Col::new(0., 0., 0.)), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key4)) => {
                if let None = ctx.state {
                    let node = gen.add(EdgeDetect::new(0.5, EdgeDetectType::Sobel), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key5)) => {
                if let None = ctx.state {
                    let node = gen.add(Noise::new(0, 2, 2), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key6)) => {
                if let None = ctx.state {
                    let node = gen.add(VoronoiNoise::new(0, 10, 10, 1., 1.), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key7)) => {
                if let None = ctx.state {
                    let node = gen.add(Select::new(0.5), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            KeyboardInput(Pressed, _, Some(Key::Key8)) => {
                if let None = ctx.state {
                    let node = gen.add(Invert::new(), Node::new(ctx.mouse_pos));
                    ctx.selected = Some(Selection::Node(node));
                }
            },
            MouseWheel(MouseScrollDelta::LineDelta(_, y)) => {
                ctx.zoom += y * ctx.zoom.sqrt();
                if ctx.zoom < 0.0001 {
                    ctx.zoom = 0.0001;
                }
            },
            MouseInput(Pressed, Mouse::Middle) => {
                if let None = ctx.state {
                    if let Some(Selection::Node(n)) = find_selected(display, &rctx, &gen, &ctx) {
                        ctx.selected = None;
                        gen.remove(&n);
                    }
                }
            },
            MouseInput(Pressed, Mouse::Left) => {
                let new_selected = find_selected(display, &rctx, &gen, &ctx);
                if let Some(Writing) = ctx.state {
                    if let Some(selected) = ctx.selected {
                        let mut same = false;
                        if let Selection::Setting(n, i) = selected {
                            if let Some(Selection::Setting(m, j)) = new_selected {
                                if n == m && i == j {
                                    same = true;
                                }
                            }
                        }
                        if same {
                            // TODO: Update caret
                        } else {
                           ctx.text.clear();
                           ctx.state = None;
                           if let Some(n) = selected.node() {
                               ctx.selected = Some(Selection::Node(n));
                           }
                        }
                    }
                }
                if let None = ctx.state {
                    match new_selected {
                        n @ Some(Selection::Node(_)) => {
                            ctx.selected = n;
                            ctx.state = Some(Dragging);
                        },
                        n @ Some(Selection::Output(..)) => {
                            ctx.selected = n;
                            ctx.state = Some(AddingEdge);
                        },
                        Some(Selection::Input(port)) => {
                            if let Some(port) = gen.disconnect(port) {
                                ctx.selected = Some(Selection::Output(port));
                                ctx.state = Some(AddingEdge);
                            }
                        },
                        Some(Selection::Setting(n, i)) => {
                            let (node, _) = gen.get(n).expect("Selected node didn't exist.");
                            ctx.text = node.setting(node.settings()[i]).to_string();
                            ctx.caret = ctx.text.len();
                            ctx.selected = Some(Selection::Setting(n, i));
                            ctx.state = Some(Writing);
                        },
                        Some(Selection::Choice(n, i, j)) => {
                            let mut node = gen.get_process_mut(n).expect("Selected node didn't exist.");
                            let setting = node.settings()[i];
                            if let SettingMut::Blend(t) = node.setting_mut(setting) {
                                *t = BlendType::iter_variants().skip(j).next().unwrap();
                            }
                            ctx.selected = Some(Selection::Node(n));
                        },
                        _ => {}
                    }
                }
            },
            MouseInput(Released, Mouse::Left) => {
                match ctx.state {
                    Some(Dragging) => {
                        ctx.state = None;
                    },
                    Some(AddingEdge) => {
                        if let Some(Selection::Output(src)) = ctx.selected {
                            if let Some(Selection::Input(trg)) = find_selected(display, &rctx, &gen, &ctx) {
                                gen.connect(src, trg);
                            }
                        }
                        ctx.state = None;
                    },
                    _ => {}
                }
            },
            MouseMoved((x, y)) => {
                ctx.mouse_window_pos = [x, y];
            }
            _ => {},
        }
    }
}

enum DecodeError {
    ParseFloatError(ParseFloatError),
    TooManyComponents,
}

impl From<ParseFloatError> for DecodeError {
    fn from(err: ParseFloatError) -> Self {
        DecodeError::ParseFloatError(err)
    }
}

fn decode_color(s: &str) -> Result<Srgb, DecodeError> {
    let input = s.split(",").collect::<Vec<_>>();
    if input.len() < 4 {
        return Err(DecodeError::TooManyComponents);
    }
    Ok(Srgb::with_alpha(
        try!(input[0].trim().parse()),
        try!(input[1].trim().parse()),
        try!(input[2].trim().parse()),
        try!(input[3].trim().parse())))
}

// TODO: Get rid of these magic numbers aka understand why you need them.
fn find_selected(display: &Display, rctx: &RenderContext, gen: &Generator<Node>, ctx: &SimContext) -> Option<Selection> {
    let dims = display.get_framebuffer_dimensions();
    let mouse_pos = ctx.mouse_pos;
    if let Some(s) = ctx.selected {
        if let Some(i) = s.node() {
            let n = gen.get(i).unwrap().0;
            let settings = n.settings();
            let size = 23.;

            if let Selection::Setting(_, j) = s {
                let pos = Vect::new(-0.5, -0.5 + (j as f32) / 20. * 0.5);
                let mut string = settings[j].to_string();
                string.push_str(": ");
                match n.setting(settings[j]) {
                    Setting::Blend(_) => {
                        let bb = rctx.fonts.bounding_box("anka", size, &string).unwrap();
                        let max = from_window_to_screen(dims, [bb.max.x, bb.max.y]);
                        let pos = pos + Vect::new(max.x * 1.25, 1. / 20. * 0.5);
                        for (choice, blend) in BlendType::iter_variants().enumerate() {
                            let blend = format!("{:?}", blend);
                            let pos = pos + Vect::new(0., (choice as f32 / 20.) * 0.5);
                            let pos = from_screen_to_world(rctx.cam, pos);
                            if let Some(bb) = rctx.fonts.bounding_box("anka", size, &blend) {
                                let min = from_screen_to_world(rctx.cam, from_window_to_screen(dims, [bb.min.x, bb.min.y]));
                                let max = from_screen_to_world(rctx.cam, from_window_to_screen(dims, [bb.max.x, bb.max.y]));
                                if pos.x + min.x < mouse_pos.x && mouse_pos.x < pos.x + max.x {
                                    if pos.y + min.y < mouse_pos.y && mouse_pos.y < pos.y + max.y {
                                        return Some(Selection::Choice(i, j, choice));
                                    }
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }

            for (j, setting) in settings.iter().enumerate() {
                let pos = Vect::new(-0.5, -0.5 + (j as f32) / 20. * 0.5);
                let pos = from_screen_to_world(rctx.cam, pos);
                let mut text = setting.to_string();
                text.push_str(": ");
                text.push_str(&n.setting(setting).to_string());
                if let Some(bb) = rctx.fonts.bounding_box("anka", size, &text) {
                    let min = from_screen_to_world(rctx.cam, from_window_to_screen(dims, [bb.min.x, bb.min.y]));
                    let max = from_screen_to_world(rctx.cam, from_window_to_screen(dims, [bb.max.x, bb.max.y]));
                    if pos.x + min.x < mouse_pos.x && mouse_pos.x < pos.x + max.x {
                        if pos.y + min.y < mouse_pos.y && mouse_pos.y < pos.y + max.y {
                            return Some(Selection::Setting(i, j));
                        }
                    }
                }
            }
        }
    }
    gen.iter()
        .enumerate()
        .map(|(i, (_, d))| (NodeIndex::new(i), d))
        .filter_map(|(i, d)| {
            if is_inside_square(mouse_pos, d.pos, ctx.node_width) {
                return Some(Selection::Node(i));
            }
            for (s, p) in d.outputs.borrow().iter().enumerate() {
                let pos = d.pos + *p;
                if is_inside_square(mouse_pos, pos, ctx.thingy_size) {
                    return Some(Selection::Output(port(i, s as u32)));
                }
            }
            for (t, p) in d.inputs.borrow().iter().enumerate() {
                let pos = d.pos + *p;
                if is_inside_square(mouse_pos, pos, ctx.thingy_size) {
                    return Some(Selection::Input(port(i, t as u32)));
                }
            }
            None
        })
        .next()
}

fn is_inside_square(pos: Vect, sqr_mid: Vect, width: f32) -> bool {
    sqr_mid.x - width / 2. < pos.x &&
    pos.x < sqr_mid.x + width / 2. &&
    sqr_mid.y - width / 2. < pos.y &&
    pos.y < sqr_mid.y + width / 2.
}
