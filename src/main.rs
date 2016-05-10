#[macro_use]
extern crate glium;
extern crate daggy;
extern crate rusttype;
extern crate arrayvec;
extern crate unicode_normalization;
extern crate texturegen;
extern crate webweaver;
extern crate nalgebra;
extern crate image;

use std::cell::RefCell;

use glium::{DisplayBuild, Program};
use glium::glutin::WindowBuilder;

use daggy::NodeIndex;

use texturegen::{Col, Generator, Port, port};
use texturegen::process::{Process, Stripes, BlendType};
use texturegen::process::Blend as BlendProcess;

use State::*;
use math::*;
use graphics::RenderContext;

mod math;
mod events;
mod graphics;

pub type Vect = nalgebra::Vec2<f32>;
pub type Mat = nalgebra::Mat4<f32>;

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
    Choice(NodeIndex, usize, usize),
}

impl Selection {
    fn node(self) -> Option<NodeIndex> {
        use Selection::*;
        match self {
            Node(node) |
            Input(Port{node, ..}) |
            Output(Port{node, ..}) |
            Setting(node, _) |
            Choice(node, _, _) =>
            Some(node),
        }
    }
}

pub struct Node {
    pos: Vect,
    shader: RefCell<Option<Program>>,
    inputs: RefCell<Vec<Vect>>,
    outputs: RefCell<Vec<Vect>>,
}

impl Node {
    fn new(pos: Vect) -> Node {
        Node {
            pos: pos,
            shader: RefCell::new(None),
            inputs: RefCell::new(vec![]),
            outputs: RefCell::new(vec![]),
        }
    }
}

pub struct SimContext {
    running: bool,
    caret: usize,
    zoom: f32,
    selected: Option<Selection>,
    state: Option<State>,
    text: String,
    mouse_pos: Vect,
    mouse_window_pos: [i32; 2],
    thingy_size: f32,
    node_width: f32,
}

impl SimContext {
    fn new() -> SimContext {
        SimContext {
            caret: 0,
            zoom: 200.,
            running: true,
            selected: None,
            state: None,
            text: String::new(),
            mouse_window_pos: [0; 2],
            mouse_pos: Vect::new(0., 0.),
            node_width: 1.,
            thingy_size: 0.1,
        }
    }
}

fn main() {
    let display = WindowBuilder::new()
        .with_vsync()
        .build_glium()
        .unwrap();
    let mut rctx = RenderContext::new(&display);
    let mut ctx = SimContext::new();
    let mut gen = Generator::new();
    construct_example_texture(&mut gen);
    while ctx.running {
        let dims = display.get_framebuffer_dimensions();
        rctx.cam = matrix(
            [[ctx.zoom / dims.0 as f32, 0., 0., 0.],
             [0., ctx.zoom / dims.1 as f32, 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]
        );

        let mouse_pos1 = from_window_to_screen(dims, ctx.mouse_window_pos);
        ctx.mouse_pos = from_screen_to_world(rctx.cam, mouse_pos1 - Vect::new(0.5, 0.5));
        if let Some(Dragging) = ctx.state {
            if let Some(Selection::Node(n)) = ctx.selected {
                if let Some(data) = gen.get_data_mut(n) {
                    data.pos = ctx.mouse_pos;
                }
            }
        }

        events::handle(&display, &rctx, &mut gen, &mut ctx);
        let gen = gen.view(|source, data, process| {
            let half_node = ctx.node_width / 2.;
            let update = |things: &mut Vec<_>, amount, dir| {
                things.clear();
                for i in 0..amount {
                    let percent = (i + 1) as f32 / (amount + 1) as f32;
                    things.push(Vect::new(
                        -half_node + percent * ctx.node_width,
                        dir * (half_node + ctx.thingy_size / 2.)));
                }
            };
            update(&mut data.inputs.borrow_mut(), process.max_in(), -1.);
            update(&mut data.outputs.borrow_mut(), process.max_out(), 1.);

            let program = Program::from_source(&display, &source.vertex, &source.fragment, None)
                .expect("Building generated shader failed");
            *data.shader.borrow_mut() = Some(program);
        });
        graphics::renderer::render(&display, &mut rctx, gen, &ctx);
    }
}

fn construct_example_texture(gen: &mut Generator<Node>) {
    let n1 = gen.add(Stripes::new(8, 1, Col::new(1., 0.5, 0.), Col::new(0.5, 0.0, 0.5)), Node::new(Vect::new(-2., -2.)));
    let n2 = gen.add(Stripes::new(1, 4, Col::new(0., 0.5, 1.), Col::new(0.0, 0.33, 0.0)), Node::new(Vect::new(0., -2.)));
    let n3 = gen.add(Stripes::new(16, 16, Col::new(0.1, 0.1, 0.2), Col::new(0.8, 0.9, 0.9)), Node::new(Vect::new(2., -2.)));
    let n4 = gen.add(BlendProcess::new(BlendType::Add, BlendType::Normal), Node::new(Vect::new(2., 0.)));
    gen.connect(port(n1, 0), port(n4, 0));
    gen.connect(port(n3, 0), port(n4, 1));
    let n5 = gen.add(BlendProcess::new(BlendType::Hard, BlendType::Normal), Node::new(Vect::new(-2., 0.)));
    gen.connect(port(n1, 0), port(n5, 0));
    gen.connect(port(n2, 0), port(n5, 1));
    let n6 = gen.add(BlendProcess::new(BlendType::Screen, BlendType::Normal), Node::new(Vect::new(0., 2.)));
    gen.connect(port(n4, 0), port(n6, 1));
    gen.connect(port(n5, 0), port(n6, 0));
}

fn input_pos(gen: &Generator<Node>, input: Port<u32>, _size: f32) -> Vect {
    let node = gen.get(input.node).unwrap();
    let pos = node.1.pos;
    let percent = (input.port + 1) as f32 / (node.0.max_in() + 1) as f32;
    Vect::new(pos[0] - 0.5 + percent, -(pos[1] - 0.5))
}

fn output_pos(gen: &Generator<Node>, output: Port<u32>, size: f32) -> Vect {
    let node = gen.get(output.node).unwrap();
    let pos = node.1.pos;
    let percent = (output.port + 1) as f32 / (node.0.max_out() + 1) as f32;
    Vect::new(pos[0] - 0.5 + percent, -(pos[1] + 0.5 + size))
}
