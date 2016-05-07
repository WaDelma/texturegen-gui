use glium::{Frame, VertexBuffer, Blend, Surface};
use glium::backend::glutin_backend::GlutinFacade;
use glium::index::{IndexBuffer, PrimitiveType};
use glium::draw_parameters::{DrawParameters};
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;
use glium::uniforms::{Uniforms, UniformsStorage, AsUniformValue};

use nalgebra::Norm;
use texturegen::GeneratorView;
use texturegen::process::{Process, Setting, BlendType};

use {SimContext, Selection, Node, Vect, input_pos, output_pos};
use super::{RenderContext, Vertex, vert};
use State::*;
use math::*;

pub fn render(display: &GlutinFacade, rctx: &mut RenderContext, gen: GeneratorView<Node>, ctx: &SimContext) {
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
    let dims = display.get_framebuffer_dimensions();
    for (_, data) in gen.iter() {
        let pos = flip_y(data.pos);
        let corner_pos = pos - Vect::new(ctx.node_width, ctx.node_width) * 0.5;
        let matrix = rctx.cam * translation(corner_pos.x, corner_pos.y);
        let uniforms = uniform! {
            matrix: *matrix.as_ref(),
        };
        draw(&mut target, &rctx, "back", "plain", &uniforms, &draw_params);
        let matrix = rctx.cam * translation(corner_pos.x + 0.05, corner_pos.y + 0.05) * scale(0.9, 0.9);
        let program = data.shader.borrow();
        let program = program.as_ref().expect("Node didn't have shader.");
        let uniforms = uniform! {
            matrix: *matrix.as_ref(),
        };
        let model = rctx.models.get("node").unwrap();
        target.draw(&model.vertices, &model.indices, &program, &uniforms, &draw_params).expect("Drawing node failed.");

        let mut draw = |things: &[_]| {
            for p in things {
                let p = pos + flip_y(*p) - Vect::new(ctx.thingy_size, ctx.thingy_size) * 0.5;
                let matrix = rctx.cam * translation(p.x, p.y) * scale(ctx.thingy_size, ctx.thingy_size);
                let uniforms = uniform! {
                    matrix: *matrix.as_ref(),
                };
                draw(&mut target, &rctx, "node", "plain", &uniforms, &draw_params);
            }
        };
        draw(&data.outputs.borrow());
        draw(&data.inputs.borrow());
    }
    let mut lines = Vec::with_capacity(gen.connections());
    if let Some(AddingEdge) = ctx.state {
        if let Some(Selection::Output(trg)) = ctx.selected {
            let src = output_pos(&gen, trg, ctx.thingy_size);
            let trg = Vect::new(ctx.mouse_pos.x, -ctx.mouse_pos.y);
            add_arrow(&mut lines, src, trg, 0.1, 0.1 * TAU);
        }
    }
    for (src, trg) in gen.iter_connections() {
        let src = output_pos(&gen, src, ctx.thingy_size);
        let trg = input_pos(&gen, trg, ctx.thingy_size);
        let trg = Vect::new(trg[0], trg[1] + ctx.thingy_size);
        add_arrow(&mut lines, src, trg, 0.1, 0.1 * TAU);
    }
    let vertices = VertexBuffer::new(display, &lines).unwrap();
    let indices = (0..lines.len() as u32).collect::<Vec<_>>();
    let indices = IndexBuffer::new(display, PrimitiveType::LinesList, &indices).unwrap();
    let matrix = rctx.cam * translation(0., 0.);
    let uniforms = uniform! {
        matrix: *matrix.as_ref(),
    };
    let program = rctx.programs.get("plain").unwrap();
    target.draw(&vertices, &indices, program, &uniforms, &draw_params).unwrap();

    if let Some(s) = ctx.selected {
        if let Some(node) = s.node() {
            let set = if let Selection::Setting(_, s) = s {
                Some(s)
            } else {
                None
            };
            let node = gen.get(node).expect("Selection should always point to real node.").0;
            let settings = node.settings();
            let size = 23.;
            for (i, setting) in settings.iter().enumerate() {
                if set == Some(i) {
                    continue;
                }
                let pos = Vect::new(0., -(i as f32) / 20.);
                let mut string = setting.to_string();
                string.push_str(": ");
                string.push_str(&node.setting(setting).to_string());
                rctx.fonts.draw_text(&display, &mut target, "anka", size, [0., 0., 0., 1.], pos, &string);
            }
            if let Some(i) = set {
                let pos = Vect::new(0., -(i as f32) / 20.);
                let mut string = settings[i].to_string();
                string.push_str(": ");
                match node.setting(settings[i]) {
                    Setting::Blend(ref b) => {
                        let bb = rctx.fonts.bounding_box("anka", size, &string).unwrap();
                        let max = from_screen_to_world(rctx.cam, from_window_to_screen(dims, [bb.max.x, bb.max.y]));
                        string.push_str(&format!("{:?}", b));
                        let pos = pos + Vect::new(max.x - 0.5, -1. / 20.);
                        let mut i = 0;
                        for blend in BlendType::iter_variants() {
                            if blend == **b {
                                continue;
                            }
                            let blend = format!("{:?}", blend);
                            let pos = pos + Vect::new(0., -(i as f32 / 20.));
                            rctx.fonts.draw_text(&display, &mut target, "anka", size, [0., 0., 0., 1.], pos, &blend);
                            i += 1;
                        }
                    },
                    _ => {
                        if let Some(Writing) = ctx.state {
                            let ch = if 0 % 120 < 60 { //TODO: Fix Me
                                '|'
                            } else {
                                ' '
                            };
                            if ctx.text.is_empty() {
                                string.push(ch);
                            } else {
                                let (a, b) = ctx.text.split_at(ctx.caret);
                                string.push_str(&format!("{}{}{}", a, ch, b));
                            }
                        }
                    }
                }
                rctx.fonts.draw_text(&display, &mut target, "anka", size, [0., 0., 0., 1.], pos, &string);
            }
        }
    }
    target.finish().unwrap();
}

fn draw<A, B>(target: &mut Frame, rctx: &RenderContext, model: &str, program: &str, uniforms: &UniformsStorage<A, B>, draw_params: &DrawParameters)
    where A: AsUniformValue,
          B: Uniforms,
{
    let model = rctx.models.get(model).unwrap();
    let program = rctx.programs.get(program).unwrap();
    target.draw(&model.vertices, &model.indices, program, uniforms, draw_params).unwrap();
}

fn add_arrow(lines: &mut Vec<Vertex>, src: Vect, trg: Vect, len: f32, theta: f32) {
    lines.push(vert(src.x, src.y));
    lines.push(vert(trg.x, trg.y));
    let vec = (src - trg).normalize();
    let mut add_part = |theta: f32| {
        let (sin, cos) = theta.sin_cos();
        let arrow = Vect::new(
            vec.x * cos - vec.y * sin,
            vec.x * sin + vec.y * cos);
        lines.push(vert(trg.x, trg.y));
        let v = trg + (arrow * len);
        lines.push(vert(v.x, v.y));
    };
    add_part(theta);
    add_part(-theta);
}
