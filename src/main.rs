#[macro_use]
extern crate glium;
extern crate daggy;
extern crate vecmath;
extern crate rusttype;
extern crate arrayvec;
extern crate unicode_normalization;
extern crate texturegen;
extern crate webweaver;
extern crate nalgebra;

use std::cell::RefCell;
use std::path::PathBuf;
use std::fs::File;
use std::io::Read;

use glium::{VertexBuffer, DisplayBuild, Blend, Program, Surface};
use glium::glutin::WindowBuilder;
use glium::index::{IndexBuffer, PrimitiveType};
use glium::draw_parameters::{DrawParameters};
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;

use vecmath::*;

use daggy::{Walker, NodeIndex};
use daggy::petgraph::EdgeDirection;

use rusttype::FontCollection;

use texturegen::{TextureGenerator, Port, port};
use texturegen::process::{Process, Constant, Stripes, BlendType};
use texturegen::process::Blend as BlendProcess;

use State::*;
use fonts::Fonts;
use math::*;

mod fonts;
mod math;

const TAU: f32 = 2. * ::std::f32::consts::PI;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
}

fn vert(x: f32, y: f32) -> Vertex {
    Vertex {
        position: [x, y],
        tex_coords: [0., 0.],
    }
}

fn vertex(x: f32, y: f32, u: f32, v: f32) -> Vertex {
    Vertex {
        position: [x, y],
        tex_coords: [u, v],
    }
}

implement_vertex!(Vertex, position, tex_coords);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Dragging,
    AddingEdge,
    Writing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    Node(NodeIndex),
    Input(Port<u32>),
    Output(Port<u32>),
    Setting(NodeIndex, usize),
}

struct Node {
    pos: [f32; 2],
    shader: RefCell<Option<Program>>,
}

impl Node {
    fn new(pos: [f32; 2]) -> Node {
        Node {
            pos: pos,
            shader: RefCell::new(None),
        }
    }
}

fn main() {
    let display = WindowBuilder::new().build_glium().unwrap();

    let path = PathBuf::from("assets")
        .join("fonts")
        .join("anka")
        .join("bold")
        .with_extension("ttf");
    let mut file = File::open(path).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    let font_data = &buffer[..];

    let mut fonts = Fonts::new(&display);
    let font = fonts.register(FontCollection::from_bytes(font_data).into_font().unwrap());

    let mut mouse_window_pos = [0; 2];

    let mut gen = TextureGenerator::<Node>::new();
    gen.register_shader_listener(|gen, node, source, event_type| {
        use texturegen::EventType::*;
        match event_type {
            Added | Changed => {
                let program = Program::from_source(&display, &source.vertex, &source.fragment, None).expect("Building generated shader failed");
                *gen.get(node).unwrap().1.shader.borrow_mut() = Some(program);
            },
            _ => {}
        }
    });
    let n1 = gen.add(Constant::new([1., 0., 0., 1.]), Node::new([-2., -2.]));
    let n2 = gen.add(Constant::new([0., 1., 0., 1.]), Node::new([0., -2.]));
    let n3 = gen.add(Constant::new([0., 0., 1., 1.]), Node::new([2., -2.]));
    let n4 = gen.add(BlendProcess::new(BlendType::Hard, BlendType::Screen), Node::new([2., 0.]));
    gen.connect(port(n1, 0), port(n4, 0));
    gen.connect(port(n3, 0), port(n4, 1));
    let n5 = gen.add(BlendProcess::new(BlendType::Soft, BlendType::Normal), Node::new([-2., 0.]));
    gen.connect(port(n1, 0), port(n5, 0));
    gen.connect(port(n2, 0), port(n5, 1));
    let n6 = gen.add(BlendProcess::new(BlendType::Screen, BlendType::Normal), Node::new([0., 2.]));
    gen.connect(port(n4, 0), port(n6, 1));
    gen.connect(port(n5, 0), port(n6, 0));

    let mut selected = None;
    let mut state = None;
    let mut text = String::new();
    let node_model = {
        let (vertices, indices) = (
            [vertex(0., 0., 0., 0.), vertex(0., 1., 0., 1.), vertex(1., 0., 1., 0.), vertex(1., 1., 1., 1.)],
            [0u32, 1, 2, 1, 2, 3]);
        (VertexBuffer::new(&display, &vertices).unwrap(),
        IndexBuffer::new(&display, PrimitiveType::TrianglesList, &indices).unwrap())
    };
    let thingy_model = &node_model;

    let back_model = {
        let (vertices, indices) = rounded_rectangle((1., 1.), (0.05, 0.05, 0.05, 0.05));
        (VertexBuffer::new(&display, &vertices).unwrap(),
        IndexBuffer::new(&display, PrimitiveType::TrianglesList, &indices).unwrap())
    };

    let thingy_size = 0.1;

    let line_program = program!(
        &display,
        140 => {
            vertex: "
                #version 140
                in vec2 position;
                uniform mat4 matrix;
                void main() {
                    gl_Position = matrix * vec4(position, 0, 1);
                }
            ",
            fragment: "
                #version 140
                out vec4 color;
                void main() {
                    color = vec4(1);
                }
            "
        }).unwrap();
    let thingy_program = &line_program;
    let back_program = &line_program;
    let mut caret = 0;
    let mut zoom = 200.;
    let mut running = true;
    let mut ticks = 0;
    while running {
        ticks += 1;
        let (w, h) = display.get_framebuffer_dimensions();
        let cam = [[zoom / w as f32, 0., 0., 0.],
                   [0., zoom / h as f32, 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]];
        let mouse_pos = from_window_to_screen((w, h), mouse_window_pos);
        let mouse_pos = from_screen_to_world(cam, [mouse_pos[0] - 0.5, mouse_pos[1] - 0.5]);
        if let Some(Dragging) = state {
            if let Some(Selection::Node(n)) = selected {
                if let Some(data) = gen.get_mut(n) {
                    data.1.pos = mouse_pos;
                }
            }
        }
        for event in display.poll_events() {
            use glium::glutin::Event::*;
            use glium::glutin::ElementState::*;
            use glium::glutin::MouseButton as Mouse;
            use glium::glutin::VirtualKeyCode as Key;
            use glium::glutin::MouseScrollDelta;
            match event {
                Closed => running = false,
                ReceivedCharacter(c) => {
                    if let Some(Writing) = state {
                        if !c.is_whitespace() && !c.is_control() {
                            if caret == text.len() {
                                text.push(c);
                            } else {
                                text.insert(caret + 1, c);
                            }
                            if caret < text.len() - 1 {
                                caret += 1;
                            }
                        }
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Right)) => {
                    if let Some(Writing) = state {
                        if caret < text.len() - 1 {
                            caret += 1;
                        }
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Left)) => {
                    if let Some(Writing) = state {
                        if caret > 0 {
                            caret -= 1;
                        }
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Back)) => {
                    if let Some(Writing) = state {
                        if !text.is_empty() {
                            text.remove(caret);
                            if caret > 0 {
                                caret -= 1;
                            }
                        }
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Escape)) => {
                    if let Some(Writing) = state {
                        text.clear();
                        selected = None;
                        state = None;
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Return)) => {
                    if let Some(Writing) = state {
                        if let Some(Selection::Setting(n, i)) = selected {
                            gen.modify(n, i, text.to_lowercase());
                        }
                        text.clear();
                        selected = None;
                        state = None;
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Tab)) => {
                    let layout = webweaver::GraphLayout::layout(gen.graph(), |_, _| 0.5, |from, to| {
                        let (edge, dir) = gen.graph().find_edge_undirected(from, to).unwrap();
                        let edge = gen.graph().edge_weight(edge).unwrap();
                        let pos = if let EdgeDirection::Outgoing = dir {
                            output_pos(&gen, port(from, edge.source), thingy_size)
                        } else {
                            input_pos(&gen, port(to, edge.target), thingy_size)
                        };
                        nalgebra::Vec2::new(pos[0], pos[1])
                    });
                    for i in 0..gen.graph().node_count() {
                        let i = NodeIndex::new(i);
                        let pos: &nalgebra::Vec2<f32> = layout.get(i).unwrap();
                        // println!("{:?}", pos);
                        gen.get_mut(i).unwrap().1.pos = *pos.as_ref();
                    }
                }
                KeyboardInput(Pressed, _, Some(Key::Key1)) => {
                    if let None = state {
                        gen.add(Constant::new([1.; 4]), Node::new(mouse_pos));
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Key2)) => {
                    if let None = state {
                        gen.add(BlendProcess::new(BlendType::Screen, BlendType::Normal), Node::new(mouse_pos));
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Key3)) => {
                    if let None = state {
                        gen.add(Stripes::new(0.25, 1., [1.; 4], [0., 0., 0., 1.]), Node::new(mouse_pos));
                    }
                },
                MouseWheel(MouseScrollDelta::LineDelta(_, y)) => {
                    zoom += y * zoom.sqrt();
                    if zoom < 0.0001 {
                        zoom = 0.0001;
                    }
                },
                MouseInput(Pressed, Mouse::Middle) => {
                    if let None = state {
                        if let Some(Selection::Node(n)) = find_selected(&gen, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom) {
                            gen.remove(&n);
                        }
                    }
                },
                MouseInput(Pressed, Mouse::Left) => {
                    let new_selected = find_selected(&gen, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom);
                    if let Some(Writing) = state {
                        let mut same = false;
                        if let Some(Selection::Setting(n, i)) = selected {
                            if let Some(Selection::Setting(m, j)) = new_selected {
                                if n == m && i == j {
                                    same = true;
                                }
                            }
                        }
                        if same {
                            // TODO: Update caret
                        } else {
                           text.clear();
                           selected = None;
                           state = None;
                        }
                    }
                    if let None = state {
                        match new_selected {
                            n @ Some(Selection::Node(_)) => {
                                selected = n;
                                state = Some(Dragging);
                            },
                            n @ Some(Selection::Output(..)) => {
                                selected = n;
                                state = Some(AddingEdge);
                            },
                            Some(Selection::Input(port)) => {
                                if let Some(port) = gen.disconnect(port) {
                                    selected = Some(Selection::Output(port));
                                    state = Some(AddingEdge);
                                }
                            },
                            Some(Selection::Setting(n, i)) => {
                                let node = gen.get(n).expect("Selected node didn't exist.");
                                text = node.0.borrow().setting(i);
                                caret =  text.len() - 1;
                                selected = Some(Selection::Setting(n, i));
                                state = Some(Writing);
                            }
                            _ => {}
                        }
                    }
                },
                MouseInput(Released, Mouse::Left) => {
                    match state {
                        Some(Dragging) => {
                            state = None;
                            selected = None;
                        },
                        Some(AddingEdge) => {
                            if let Some(Selection::Output(src)) = selected {
                                if let Some(Selection::Input(trg)) = find_selected(&gen, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom) {
                                    gen.connect(src, trg);
                                }
                            }
                            state = None;
                            selected = None;
                        },
                        _ => {}
                    }
                },
                MouseMoved((x, y)) => {
                    mouse_window_pos = [x, y];
                }
                _ => {},
            }
        }

        let mut target = display.draw();
        target.clear_color(0.0157, 0.0173, 0.0204, 1.);
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

        for (i, (process, data)) in gen.iter().enumerate() {
            let i = NodeIndex::new(i);
            let position = data.pos;
            let x = position[0];
            let y = position[1];
            let matrix = translation(x - 0.5, -y - 0.5);
            let matrix = col_mat4_mul(cam, matrix);
            let program = back_program;
            let uniforms = uniform! {
                matrix: matrix,
            };
            let vertices = &back_model.0;
            let indices = &back_model.1;
            target.draw(vertices, indices, &program, &uniforms, &draw_params).expect("Drawing background failed.");
            let matrix = translation(x - 0.5 + 0.05, -y - 0.5 + 0.05);
            let matrix = col_mat4_mul(matrix,
                             [[0.9, 0. , 0., 0.],
                              [0. , 0.9, 0., 0.],
                              [0. , 0. , 1., 0.],
                              [0. , 0. , 0., 1.]]);
            let matrix = col_mat4_mul(cam, matrix);
            let program = data.shader.borrow();
            let program = program.as_ref().expect("Node didn't have shader.");
            let uniforms = uniform! {
                matrix: matrix,
            };
            let vertices = &node_model.0;
            let indices = &node_model.1;
            target.draw(vertices, indices, &program, &uniforms, &draw_params).expect("Drawing node failed.");
            let settings = process.borrow().settings();
            for (ii, setting) in settings.iter().enumerate() {
                let size = 0.1 * zoom;
                let pos = [x + 0.45, y - 0.5 + ((ii + 1) as f32 / (settings.len() + 1) as f32)];
                let pos = from_world_to_screen(cam, pos);
                let mut string = setting.clone();
                string.push_str(": ");
                let mut flag = true;
                if let Some(Writing) = state {
                    if let Some(Selection::Setting(n, j)) = selected {
                        if n == i && ii == j {
                            let ch = if ticks % 120 < 60 {
                                '|'
                            } else {
                                ' '
                            };
                            if text.is_empty() {
                                string.push(ch);
                            } else {
                                let (a, b) = text.split_at(caret + 1);
                                string.push_str(&format!("{}{}{}", a, ch, b));
                            }
                            flag = false;
                        }
                    }
                }
                if flag {
                    string.push_str(&process.borrow().setting(ii));
                }
                fonts.draw_text(&display, &mut target, font, size, [0., 0., 0., 1.], pos, &string);
            }
            for s in 0..process.borrow().max_out() {
                let pos = output_pos(&gen, port(i, s), thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let matrix = translation(x, y);
                let matrix = col_mat4_mul(matrix, scale(thingy_size, thingy_size));
                let matrix = col_mat4_mul(cam, matrix);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                let vertices = &thingy_model.0;
                let indices = &thingy_model.1;
                let program = thingy_program;
                target.draw(vertices, indices, &program, &uniforms, &draw_params).unwrap();
            }

            for t in 0..process.borrow().max_in() {
                let pos = input_pos(&gen, port(i, t), thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let matrix = translation(x, y);
                let matrix = col_mat4_mul(matrix, scale(thingy_size, thingy_size));
                let matrix = col_mat4_mul(cam, matrix);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                let vertices = &thingy_model.0;
                let indices = &thingy_model.1;
                let program = thingy_program;
                target.draw(vertices, indices, &program, &uniforms, &draw_params).unwrap();
            }
        }
        let mut lines = Vec::with_capacity(gen.connections());
        if let Some(AddingEdge) = state {
            if let Some(Selection::Output(trg)) = selected {
                let src = output_pos(&gen, trg, thingy_size);
                let trg = [mouse_pos[0], -mouse_pos[1]];
                add_arrow(&mut lines, src, trg, 0.1, 0.1 * TAU);
            }
        }
        for (src, trg) in gen.iter_connections() {
            let src = output_pos(&gen, src, thingy_size);
            let trg = input_pos(&gen, trg, thingy_size);
            let trg = [trg[0], trg[1] + thingy_size];
            add_arrow(&mut lines, src, trg, 0.1, 0.1 * TAU);
        }
        let vertices = VertexBuffer::new(&display, &lines).unwrap();
        let indices = (0..lines.len() as u32).collect::<Vec<_>>();
        let indices = IndexBuffer::new(&display, PrimitiveType::LinesList, &indices).unwrap();
        let matrix = translation(0., 0.);
        let matrix = col_mat4_mul(cam, matrix);
        let uniforms = uniform! {
            matrix: matrix,
        };
        let program = &line_program;
        target.draw(&vertices, &indices, program, &uniforms, &draw_params).unwrap();

        target.finish().unwrap();
    }
}

fn input_pos(gen: &TextureGenerator<Node>, input: Port<u32>, _size: f32) -> [f32; 2] {
    let node = gen.get(input.node).unwrap();
    let process = node.0.borrow();
    let pos = node.1.pos;
    [pos[0] - 0.5 + ((input.port + 1) as f32 / (process.max_in() + 1) as f32),
    -(pos[1] - 0.5)]
}

fn output_pos(gen: &TextureGenerator<Node>, output: Port<u32>, size: f32) -> [f32; 2] {
    let node = gen.get(output.node).unwrap();
    let process = node.0.borrow();
    let pos = node.1.pos;
    [pos[0] - 0.5 + ((output.port + 1) as f32 / (process.max_out() + 1) as f32),
    -(pos[1] + 0.5 + size)]
}

fn add_arrow(lines: &mut Vec<Vertex>, src: [f32; 2], trg: [f32; 2], len: f32, theta: f32) {
    lines.push(vert(src[0], src[1]));
    lines.push(vert(trg[0], trg[1]));
    let len = [len, len];
    let vec = vec2_normalized_sub(src, trg);
    let cs = theta.cos();
    let sn = theta.sin();
    let arrow = [vec[0] * cs - vec[1] * sn, vec[0] * sn + vec[1] * cs];
    lines.push(vert(trg[0], trg[1]));
    let v = vec2_add(trg, vec2_mul(arrow, len));
    lines.push(vert(v[0], v[1]));

    let vec = vec2_normalized_sub(src, trg);
    let cs = (-theta).cos();
    let sn = (-theta).sin();
    let arrow = [vec[0] * cs - vec[1] * sn, vec[0] * sn + vec[1] * cs];
    lines.push(vert(trg[0], trg[1]));
    let v = vec2_add(trg, vec2_mul(arrow, len));
    lines.push(vert(v[0], v[1]));
}

fn find_selected(gen: &TextureGenerator<Node>, mouse_pos: [f32; 2], cam: [[f32; 4]; 4], (w, h): (u32, u32), size: f32, fonts: &Fonts, font: usize, zoom: f32) -> Option<Selection> {
    gen.iter()
        .enumerate()
        .filter_map(|(i, (n, d))| {
            let i = NodeIndex::new(i);
            let pos = d.pos;
            if pos[0] - 0.5 < mouse_pos[0] && mouse_pos[0] < pos[0] + 0.5 {
                if pos[1] - 0.5 < mouse_pos[1] && mouse_pos[1] < pos[1] + 0.5 {
                    return Some(Selection::Node(i));
                }
            }
            let settings = n.borrow().settings();
            for (j, setting) in settings.iter().enumerate() {
                let pos = [pos[0] + 0.45, pos[1] - 0.5 + ((j + 1) as f32 / (settings.len() + 1) as f32)];
                let size = 0.1 * zoom;
                let mut text = setting.clone();
                text.push_str(": ");
                text.push_str(&n.borrow().setting(j));
                if let Some(bb) = fonts.bounding_box(font, size, &text) {
                    let min = from_screen_to_world(cam, from_window_to_screen((w, h), [bb.min.x, bb.min.y]));
                    let max = from_screen_to_world(cam, from_window_to_screen((w, h), [bb.max.x, bb.max.y]));
                    if pos[0] + min[0] < mouse_pos[0] && mouse_pos[0] < pos[0] + max[0] {
                        if pos[1] + min[1] < mouse_pos[1] && mouse_pos[1] < pos[1] + max[1] {
                            return Some(Selection::Setting(i, j));
                        }
                    }
                }
            }
            for s in 0..n.borrow().max_out() {
                let pos = output_pos(&gen, port(i, s), size);
                let pos = [pos[0], -pos[1]];
                if pos[0] - size < mouse_pos[0] && mouse_pos[0] < pos[0] + size {
                    if pos[1] - size < mouse_pos[1] && mouse_pos[1] < pos[1] + size {
                        return Some(Selection::Output(port(i, s)));
                    }
                }
            }
            for t in 0..n.borrow().max_in() {
                let pos = input_pos(&gen, port(i, t), size);
                let pos = [pos[0], -pos[1]];
                if pos[0] - size < mouse_pos[0] && mouse_pos[0] < pos[0] + size {
                    if pos[1] - size < mouse_pos[1] && mouse_pos[1] < pos[1] + size {
                        return Some(Selection::Input(port(i, t)));
                    }
                }
            }
            None
        })
        .next()
}

fn rounded_rectangle((width, height): (f32, f32), (tlr, trr, blr, brr): (f32, f32, f32, f32)) -> (Vec<Vertex>, Vec<u32>) {
    fn do_corner(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, cur: f32, (x, y): (f32, f32), (a, b, c): (u32, u32, u32), angle: f32) {
        if cur > 0. {
            let num_sides = (0.25 * cur).max(1.);
            for i in 0..num_sides as usize {
                let i = i + 1;
                let radians = (i as f32 / (num_sides + 1.)) * 0.25 * TAU + angle;
                let sin = radians.sin();
                let cos = radians.cos();
                let x = x + sin * cur;
                let y = y - cos * cur;

                vertices.push(vert(x, y));

                let d = vertices.len() as u32 - 1;
                if i == 1  {
                    indices.extend(&[a, b, d][..]);
                } else {
                    indices.extend(&[a, d - 1, d][..]);
                }

                if i == num_sides as usize {
                    indices.extend(&[a, d, c][..]);
                }
            }
        }
    }
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half = 0.5 * width.min(height);
    let tlr = half.min(tlr);
    let trr = half.min(trr);
    let blr = half.min(blr);
    let brr = half.min(brr);

    vertices.push(vert(tlr, 0. ));
    vertices.push(vert(tlr, tlr));
    vertices.push(vert(0. , tlr));

    vertices.push(vert(width - trr, 0. ));
    vertices.push(vert(width - trr, trr));
    vertices.push(vert(width - 0. , trr));

    vertices.push(vert(blr, height - 0. ));
    vertices.push(vert(blr, height - blr));
    vertices.push(vert(0. , height - blr));

    vertices.push(vert(width - brr, height - 0. ));
    vertices.push(vert(width - brr, height - brr));
    vertices.push(vert(width - 0. , height - brr));

    indices.extend(&[0,3,1, 1,3,4, 2,1,8, 8,1,7, 7,1,4, 7,4,10, 10,4,5, 10,5,11, 6,7,10, 6,10,9][..]);

    do_corner(&mut vertices, &mut indices, tlr, (   0. + tlr,     0. + tlr), ( 1,  2, 0), 0.75 * TAU);
    do_corner(&mut vertices, &mut indices, trr, (width - trr,     0. + trr), ( 4,  3, 5), 0.00 * TAU);
    do_corner(&mut vertices, &mut indices, brr, (width - brr, height - brr), (10, 11, 9), 0.25 * TAU);
    do_corner(&mut vertices, &mut indices, brr, (   0. + blr, height - blr), ( 7,  6, 8), 0.50 * TAU);
    (vertices, indices)
}
