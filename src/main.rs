#[macro_use]
extern crate glium;
extern crate daggy;
extern crate vecmath;
extern crate bit_set;
extern crate rusttype;
extern crate arrayvec;
extern crate unicode_normalization;

use std::borrow::Cow;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::f32;
use std::path::PathBuf;
use std::fs::File;
use std::io::Read;

use rusttype::{FontCollection, Font, Scale, point, vector, PositionedGlyph};
use rusttype::gpu_cache::{Cache};
use rusttype::Rect;

use arrayvec::ArrayVec;

use glium::{Frame, VertexBuffer, DisplayBuild, Blend, Program, Surface};
use glium::texture::{Texture2d, RawImage2d, MipmapsOption, UncompressedFloatFormat, ClientFormat};
use glium::backend::Facade;
use glium::backend::glutin_backend::GlutinFacade;
use glium::glutin::WindowBuilder;
use glium::index::{NoIndices, IndexBuffer, PrimitiveType};
use glium::draw_parameters::{DrawParameters};
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;
use glium::uniforms::{MinifySamplerFilter, MagnifySamplerFilter};

use bit_set::BitSet;

use vecmath::*;

use daggy::{Walker, NodeIndex};

use State::*;
use process::Process;
use process::inputs;
use process::combiners::{self, BlendType};
use dag::PortNumbered;
use shader::{Context, Shader};

mod dag;
mod shader;
mod process;

const TAU: f32 = 2. * ::std::f32::consts::PI;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
}

fn vert(x: f32, y: f32) -> Vertex {
    Vertex {position: [x, y]}
}

implement_vertex!(Vertex, position);

struct Node {
    process: Rc<RefCell<Process>>,
    program: RefCell<Option<Program>>,
    position: [f32; 2],
}

impl Node {
    fn new(process: Rc<RefCell<Process>>, position: [f32; 2]) -> Node {
        Node {
            process: process,
            position: position,
            program: RefCell::new(None),
        }
    }

    fn max_in(&self) -> u32 {
        self.process.borrow().max_in()
    }

    fn max_out(&self) -> u32 {
        self.process.borrow().max_out()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Dragging,
    AddingEdge,
    Writing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    Node(NodeIndex),
    Input(NodeIndex, u32),
    Output(NodeIndex, u32),
    Setting(NodeIndex, usize),
}

struct Fonts<'a> {
    dpi_factor: f32,
    cache: Cache,
    cache_tex: Texture2d,
    program: Program,
    program_bg: Program,
    fonts: HashMap<usize, Font<'a>>,
}

impl<'a> Fonts<'a> {
    fn new(display: &GlutinFacade) -> Fonts {
        let dpi_factor = display.get_window().unwrap().hidpi_factor();
        let (w, h) = display.get_framebuffer_dimensions();
        let (cache_width, cache_height) = (w * dpi_factor as u32, h * dpi_factor as u32);
        Fonts {
            fonts: HashMap::new(),
            dpi_factor: dpi_factor,
            cache: Cache::new(cache_width, cache_height, 0.1, 0.1),
            program: program!(
                display,
                140 => {
                    vertex: "
                        #version 140
                        in vec2 position;
                        in vec2 tex_coords;
                        uniform vec4 color;
                        out vec2 v_tex_coords;
                        out vec4 v_color;
                        void main() {
                            gl_Position = vec4(position, 0.0, 1.0);
                            v_tex_coords = tex_coords;
                            v_color = color;
                        }
                    ",
                    fragment: "
                        #version 140
                        uniform sampler2D tex;
                        in vec2 v_tex_coords;
                        in vec4 v_color;
                        out vec4 color;
                        void main() {
                            color = v_color * vec4(1.0, 1.0, 1.0, texture(tex, v_tex_coords).r);
                        }
                    "
                }).unwrap(),
            program_bg: program!(
                display,
                140 => {
                    vertex: "
                        #version 140
                        in vec2 position;
                        uniform vec4 color;
                        out vec4 v_color;
                        void main() {
                            gl_Position = vec4(position, 0.0, 1.0);
                            v_color = color;
                        }
                    ",
                    fragment: "
                        #version 140
                        in vec4 v_color;
                        out vec4 color;
                        void main() {
                            color = v_color;
                        }
                    "
                }).unwrap(),
            cache_tex: Texture2d::with_format(
                display,
                RawImage2d {
                    data: Cow::Owned(vec![128u8; cache_width as usize * cache_height as usize]),
                    width: cache_width,
                    height: cache_height,
                    format: ClientFormat::U8
                },
            UncompressedFloatFormat::U8,
            MipmapsOption::NoMipmap).unwrap(),
        }
    }

    pub fn register(&mut self, font: Font<'a>) -> usize {
        let id = self.fonts.len();
        self.fonts.insert(id, font);
        id
    }

    pub fn bounding_box(&self, font: usize, size: f32, text: &str) -> Option<Rect<f32>> {
        let font = self.fonts.get(&font).expect(&format!("Font with id {} didn't exist.", font));
        self.layout(font, Scale::uniform(size * self.dpi_factor), text).1
    }

    fn layout<'b>(&self, font: &'b Font, scale: Scale, text: &str) -> (Vec<PositionedGlyph<'b>>, Option<Rect<f32>>) {
        use unicode_normalization::UnicodeNormalization;
        let mut result = Vec::new();
        let metrics = font.v_metrics(scale);
        let advance_height = metrics.ascent - metrics.descent + metrics.line_gap;
        let mut caret = point(0.0, metrics.ascent - advance_height / 2.);
        let mut last_glyph = None;
        for c in text.nfc() {
            if c.is_control() {
                continue;
            }
            let cur = if let Some(g) = font.glyph(c) {
                g
            } else {
                continue;
            };
            if let Some(id) = last_glyph.take() {
                caret.x += font.pair_kerning(scale, id, cur.id());
            }
            last_glyph = Some(cur.id());
            let glyph = cur.scaled(scale).positioned(caret);
            caret.x += glyph.unpositioned().h_metrics().advance_width;
            result.push(glyph);
        }
        if result.is_empty() {
            return (result, None);
        }
        let mut bg = Rect {min: point(f32::MAX, f32::MAX), max: point(f32::MIN, f32::MIN)};
        for glyph in &result {
            if let Some(Rect{min, max}) = glyph.pixel_bounding_box() {
                bg.min.x = bg.min.x.min(min.x as f32);
                bg.min.y = bg.min.y.min(min.y as f32);
                bg.max.x = bg.max.x.max(max.x as f32);
                bg.max.y = bg.max.y.max(max.y as f32);
            }
        }
        (result, Some(bg))
    }

    fn draw_text(&mut self, display: &GlutinFacade, target: &mut Frame, font: usize, size: f32, color: [f32; 4], pos: [f32; 2], text: &str) {
        fn get_rect(min: rusttype::Point<u32>, tex: &RawImage2d<u8>) -> glium::Rect {
            glium::Rect {
                left: min.x,
                bottom: min.y,
                width: tex.width,
                height: tex.height,
            }
        }
        let (w, h) = display.get_framebuffer_dimensions();
        let (w, h) = (w as f32, h as f32);
        let origin = point(pos[0], pos[1]);
        let font_data = self.fonts.get(&font).expect(&format!("Font with id {} didn't exist.", font));
        let (glyphs, bg) = self.layout(font_data, Scale::uniform(size * self.dpi_factor), text);
        if let Some(bg) = bg {
            let min = vector(bg.min.x as f32 / w, -bg.min.y as f32 / h) - vector(0.5, -0.5);
            let max = vector(bg.max.x as f32 / w, -bg.max.y as f32 / h) - vector(0.5, -0.5);
            let pos = Rect {
                min: origin + min * 2.,
                max: origin + max * 2.,
            };
            let vertices = VertexBuffer::new(display, &[
                vert(pos.min.x, pos.max.y),
                vert(pos.min.x, pos.min.y),
                vert(pos.max.x, pos.min.y),
                vert(pos.max.x, pos.min.y),
                vert(pos.max.x, pos.max.y),
                vert(pos.min.x, pos.max.y),
            ]).unwrap();
            let uniforms = uniform! {
                color: [1f32; 4],
            };
            target.draw(&vertices,
                        NoIndices(PrimitiveType::TrianglesList),
                        &self.program_bg, &uniforms,
                        &DrawParameters {
                            blend: Blend::alpha_blending(),
                            ..Default::default()
                        }).unwrap();
        }

        for glyph in &glyphs {
            self.cache.queue_glyph(font, glyph.clone());
        }
        let cache_tex = &self.cache_tex;
        self.cache.cache_queued(|rect, data| {
            let tex = RawImage2d {
                data: Cow::Borrowed(data),
                width: rect.width(),
                height: rect.height(),
                format: ClientFormat::U8,
            };
            cache_tex.main_level().write(get_rect(rect.min, &tex), tex);
        }).unwrap();
        let uniforms = uniform! {
            tex: self.cache_tex.sampled()
                .magnify_filter(MagnifySamplerFilter::Nearest)
                .minify_filter(MinifySamplerFilter::Nearest),
            color: color,
        };
        let vertex_buffer = {
            #[derive(Copy, Clone)]
            struct Vertex {
                position: [f32; 2],
                tex_coords: [f32; 2]
            }
            fn vertex(position: [f32; 2], tex_coords: [f32; 2]) -> Vertex {
                Vertex {
                    position: position,
                    tex_coords: tex_coords,
                }
            }
            implement_vertex!(Vertex, position, tex_coords);
            let vertices = glyphs.iter().flat_map(|g| {
                if let Ok(Some((uv, screen))) = self.cache.rect_for(font, g) {
                    let min = vector(screen.min.x as f32 / w, -screen.min.y as f32 / h) - vector(0.5, -0.5);
                    let max = vector(screen.max.x as f32 / w, -screen.max.y as f32 / h) - vector(0.5, -0.5);
                    let pos = Rect {
                        min: origin + min * 2.,
                        max: origin + max * 2.,
                    };
                    ArrayVec::from([
                        vertex([pos.min.x, pos.max.y], [uv.min.x, uv.max.y]),
                        vertex([pos.min.x, pos.min.y], [uv.min.x, uv.min.y]),
                        vertex([pos.max.x, pos.min.y], [uv.max.x, uv.min.y]),
                        vertex([pos.max.x, pos.min.y], [uv.max.x, uv.min.y]),
                        vertex([pos.max.x, pos.max.y], [uv.max.x, uv.max.y]),
                        vertex([pos.min.x, pos.max.y], [uv.min.x, uv.max.y]),
                    ])
                } else {
                    ArrayVec::new()
                }
            }).collect::<Vec<_>>();
            VertexBuffer::new(display, &vertices).unwrap()
        };
        target.draw(&vertex_buffer,
                    NoIndices(PrimitiveType::TrianglesList),
                    &self.program, &uniforms,
                    &DrawParameters {
                        blend: Blend::alpha_blending(),
                        ..Default::default()
                    }).unwrap();
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
    let mut dag = PortNumbered::<Node, u32>::new();
    let n1 = dag.add_node(Node::new(inputs::Constant::new([1., 0., 0., 1.]), [-2., -2.]));
    let n2 = dag.add_node(Node::new(inputs::Constant::new([0., 1., 0., 1.]), [0., -2.]));
    let n3 = dag.add_node(Node::new(inputs::Constant::new([0., 0., 1., 1.]), [2., -2.]));
    let n4 = dag.add_node(Node::new(combiners::Blend::new(BlendType::Hard, BlendType::Screen), [2., 0.]));
    let _ = dag.update_edge(n1, 0, n4, 0).unwrap();
    let _ = dag.update_edge(n3, 0, n4, 1).unwrap();
    let n5 = dag.add_node(Node::new(combiners::Blend::new(BlendType::Soft, BlendType::Normal), [-2., 0.]));
    let _ = dag.update_edge(n1, 0, n5, 0).unwrap();
    let _ = dag.update_edge(n2, 0, n5, 1).unwrap();
    let n6 = dag.add_node(Node::new(combiners::Blend::new(BlendType::Screen, BlendType::Normal), [0., 2.]));
    let _ = dag.update_edge(n4, 0, n6, 1).unwrap();
    let _ = dag.update_edge(n5, 0, n6, 0).unwrap();
    update_dag(&display, &dag, n1);
    update_dag(&display, &dag, n2);
    update_dag(&display, &dag, n3);
    let mut selected = None;
    let mut state = None;
    let mut text = String::new();
    let node_model = {
        let (vertices, indices) = (
            [vert(0., 0.), vert(0., 1.), vert(1., 0.), vert(1., 1.)],
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
    let thingy_scale = [[thingy_size, 0., 0., 0.],
                        [0., thingy_size, 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]];

    let mut line_program = Shader::new();
    line_program.add_vertex("gl_Position = matrix * vec4(position, 0, 1);\n");
    line_program.add_fragment("color = vec4(1);\n");
    let line_program = line_program.build(&display);
    let thingy_program = &line_program;
    let back_program = &line_program;
    let mut zoom = 150.;

    let mut running = true;
    while running {
        let (w, h) = display.get_framebuffer_dimensions();
        let cam = [[zoom / w as f32, 0., 0., 0.],
                   [0., zoom / h as f32, 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]];
        let rel_x = mouse_window_pos[0] as f32 / w as f32 - 0.5;
        let rel_y = mouse_window_pos[1] as f32 / h as f32 - 0.5;
        let m = inverse_transform(cam, [rel_x, rel_y]);
        let mouse_pos = [m[0] * 2., m[1] * 2.];
        if let Some(Dragging) = state {
            if let Some(Selection::Node(o)) = selected {
                dag.node_weight_mut(o).unwrap().position = mouse_pos;
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
                            text.push(c);
                        }
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Back)) => {
                    if let Some(Writing) = state {
                        text.pop();
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
                            dag.node_weight(n)
                                .unwrap()
                                .process
                                .borrow_mut()
                                .modify(i, text.to_lowercase());
                            update_dag(&display, &dag, n);
                        }
                        text.clear();
                        selected = None;
                        state = None;
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Key1)) => {
                    if let None = state {
                        let n = dag.add_node(Node::new(inputs::Constant::new([1., 1., 1., 1.]), mouse_pos));
                        update_dag(&display, &dag, n);
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Key2)) => {
                    if let None = state {
                        let n = dag.add_node(Node::new(combiners::Blend::new(BlendType::Multiply, BlendType::Normal), mouse_pos));
                        update_dag(&display, &dag, n);
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
                        if let Some(Selection::Node(n)) = find_selected(&dag, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom) {
                            let children = dag.children(n).map(|(_, n, _)| n).collect::<Vec<_>>();
                            dag.remove_outgoing_edges(n);
                            for c in children {
                                update_dag(&display, &dag, c);
                            }
                            dag.remove_node(n);
                        }
                    }
                },
                MouseInput(Pressed, Mouse::Left) => {
                    if let None = state {
                        state = Some(AddingEdge);
                        match find_selected(&dag, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom) {
                            s @ Some(Selection::Output(..)) => {
                                selected = s;
                            },
                            Some(Selection::Input(n, i)) => {
                                let (s, i) = dag.remove_edge_to_port(n, i).unwrap();
                                update_dag(&display, &dag, n);
                                selected = Some(Selection::Output(s, i));
                            },
                            n @ Some(Selection::Setting(..)) => {
                                selected = n;
                                state = Some(Writing);
                                println!("Started writing!");
                            }
                            _ => {}
                        }
                    }
                },
                MouseInput(Released, Mouse::Left) => {
                    if let Some(AddingEdge) = state {
                        if let Some(Selection::Output(source, i)) = selected {
                            if let Some(Selection::Input(target, o)) = find_selected(&dag, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom) {
                                if let Ok(_) = dag.update_edge(source, i, target, o) {
                                    update_dag(&display, &dag, target);
                                }
                            }
                        }
                        state = None;
                        selected = None;
                    }
                },
                MouseInput(Pressed, Mouse::Right) => {
                    if let None = state {
                        match find_selected(&dag, mouse_pos, cam, (w, h), thingy_size, &fonts, font, zoom) {
                            n @ Some(Selection::Node(_)) => {
                                selected = n;
                                state = Some(Dragging);
                            },
                            _ => {}
                        }
                    }
                },
                MouseInput(Released, Mouse::Right) => {
                    if let Some(Dragging) = state {
                        state = None;
                        selected = None;
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
        for (i, node) in dag.raw_nodes().iter().enumerate() {
            let node = &node.weight;
            let x = node.position[0];
            let y = node.position[1];
            let matrix = translation(x - 0.5, -y - 0.5);
            let matrix = col_mat4_mul(cam, matrix);
            let program = back_program;
            let uniforms = uniform! {
                matrix: matrix,
            };
            let vertices = &back_model.0;
            let indices = &back_model.1;
            target.draw(vertices, indices, &program, &uniforms, &draw_params).unwrap();
            let matrix = translation(x - 0.5 + 0.05, -y - 0.5 + 0.05);
            let matrix = col_mat4_mul(matrix,
                             [[0.9, 0. , 0., 0.],
                              [0. , 0.9, 0., 0.],
                              [0. , 0. , 1., 0.],
                              [0. , 0. , 0., 1.]]);
            let matrix = col_mat4_mul(cam, matrix);
            let program = node.program.borrow();
            let program = program.as_ref().unwrap();
            let uniforms = uniform! {
                matrix: matrix,
            };
            let vertices = &node_model.0;
            let indices = &node_model.1;
            target.draw(vertices, indices, &program, &uniforms, &draw_params).unwrap();
            let settings = node.process.borrow().settings();
            for (ii, setting) in settings.iter().enumerate() {
                let size = 0.1 * zoom;
                let pos = [x + 0.45, y - 0.5 + ((ii + 1) as f32 / (settings.len() + 1) as f32)];
                let pos = transform(cam, pos);
                let pos = [pos[0] + 1., -(pos[1] + 1.)];
                let mut string = setting.clone();
                string.push_str(": ");
                let mut flag = true;
                if let Some(Writing) = state {
                    if let Some(Selection::Setting(n, j)) = selected {
                        if n.index() == i && ii == j {
                            string.push_str(&text);
                            flag = false;
                        }
                    }
                }
                if flag {
                    string.push_str(&node.process.borrow().setting(ii));
                }
                fonts.draw_text(&display, &mut target, font, size, [0., 0., 0., 1.], pos, &string);
            }
            for s in 0..node.max_out() {
                let pos = output_pos(node, s, thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let matrix = translation(x, y);
                let matrix = col_mat4_mul(matrix, thingy_scale);
                let matrix = col_mat4_mul(cam, matrix);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                let vertices = &thingy_model.0;
                let indices = &thingy_model.1;
                let program = thingy_program;
                target.draw(vertices, indices, &program, &uniforms, &draw_params).unwrap();
            }

            for t in 0..node.max_in() {
                let pos = input_pos(node, t, thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let matrix = translation(x, y);
                let matrix = col_mat4_mul(matrix, thingy_scale);
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
        let mut lines = Vec::with_capacity(dag.edge_count());
        if let Some(AddingEdge) = state {
            if let Some(Selection::Output(source, s)) = selected {
                let src = dag.node_weight(source).unwrap();
                let src = output_pos(src, s, thingy_size);
                let trg = [mouse_pos[0], -mouse_pos[1]];
                add_arrow(&mut lines, src, trg, 0.1, 0.1 * TAU);
            }
        }
        for (source, s, target, t) in dag.edges() {
            let src = dag.node_weight(source).unwrap();
            let trg = dag.node_weight(target).unwrap();
            let src = output_pos(src, s, thingy_size);
            let trg = input_pos(trg, t, thingy_size);
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

fn transform(matrix: [[f32; 4]; 4], vector: [f32; 2]) -> [f32; 2] {
    let m = col_mat4_transform(matrix, [vector[0], vector[1], 0., 1.]);
    [m[0], m[1]]
}

fn inverse_transform(matrix: [[f32; 4]; 4], vector: [f32; 2]) -> [f32; 2] {
    let m = col_mat4_transform(mat4_inv(matrix), [vector[0], vector[1], 0., 1.]);
    [m[0], m[1]]
}

fn translation(x: f32, y: f32) -> [[f32; 4]; 4] {
    [[1., 0., 0., 0.],
     [0., 1., 0., 0.],
     [0., 0., 1., 0.],
     [x , y , 0., 1.]]
}

fn update_dag<F: Facade>(facade: &F, dag: &PortNumbered<Node>, node: NodeIndex) {
    update_node(facade, dag, node);
    for (_source, node, _target) in dag.children(node) {
        update_dag(facade, dag, node);
    }
}

fn update_node<F: Facade>(facade: &F, dag: &PortNumbered<Node>, node: NodeIndex) {
    fn recurse(shader: &mut Shader, dag: &PortNumbered<Node>, node: NodeIndex, visited: &mut HashSet<NodeIndex>) {
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);
        let process = dag.node_weight(node).unwrap().process.borrow();
        for s in 0..process.max_in() {
            shader.add_fragment(format!("vec4 in_{}_{} = vec4(0, 0, 0, 0);\n", node.index(), s));
        }
        let mut inputs = BitSet::new();
        for (parent, source, target) in dag.parents(node) {
            inputs.insert(target as usize);
            recurse(shader, dag, parent, visited);
            shader.add_fragment(format!("in_{}_{} = out_{}_{};\n", node.index(), target, parent.index(), source));
        }
        let mut context = Context::new(node.index(), inputs, process.max_out());
        shader.add_fragment(process.shader(&mut context));
    }
    let mut result = Shader::new();
    result.add_vertex("gl_Position = matrix * vec4(position, 0, 1);\n");
    result.add_fragment("vec4 one = vec4(1, 1, 1, 1);\n");
    recurse(&mut result, dag, node, &mut HashSet::new());
    result.add_fragment(format!("color = out_{}_0;\n", node.index()));
    *dag.node_weight(node).unwrap().program.borrow_mut() = Some(result.build(facade));
}

fn input_pos(node: &Node, index: u32, _size: f32) -> [f32; 2] {
    [node.position[0] - 0.5 + ((index + 1) as f32 / (node.max_in() + 1) as f32),
    -(node.position[1] - 0.5)]
}

fn output_pos(node: &Node, index: u32, size: f32) -> [f32; 2] {
    [node.position[0] - 0.5 + ((index + 1) as f32 / (node.max_out() + 1) as f32),
    -(node.position[1] + 0.5 + size)]
}

fn add_arrow(lines: &mut Vec<Vertex>, src: [f32; 2], trg: [f32; 2], len: f32, theta: f32) {
    lines.push(Vertex {
        position: src,
    });
    lines.push(Vertex {
        position: trg,
    });
    let len = [len, len];
    let vec = vec2_normalized_sub(src, trg);
    let cs = theta.cos();
    let sn = theta.sin();
    let arrow = [vec[0] * cs - vec[1] * sn, vec[0] * sn + vec[1] * cs];
    lines.push(Vertex {
        position: trg,
    });
    lines.push(Vertex {
        position: vec2_add(trg, vec2_mul(arrow, len)),
    });

    let vec = vec2_normalized_sub(src, trg);
    let cs = (-theta).cos();
    let sn = (-theta).sin();
    let arrow = [vec[0] * cs - vec[1] * sn, vec[0] * sn + vec[1] * cs];
    lines.push(Vertex {
        position: trg,
    });
    lines.push(Vertex {
        position: vec2_add(trg, vec2_mul(arrow, len)),
    });
}

fn find_selected(dag: &PortNumbered<Node>, mouse_pos: [f32; 2], cam: [[f32; 4]; 4], (w, h): (u32, u32), size: f32, fonts: &Fonts, font: usize, zoom: f32) -> Option<Selection> {
    let (w, h) = (w as f32, h as f32);
    dag.raw_nodes()
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            let i = NodeIndex::new(i);
            let pos = n.weight.position;
            if pos[0] - 0.5 < mouse_pos[0] && mouse_pos[0] < pos[0] + 0.5 {
                if pos[1] - 0.5 < mouse_pos[1] && mouse_pos[1] < pos[1] + 0.5 {
                    return Some(Selection::Node(i));
                }
            }
            let settings = n.weight.process.borrow().settings();
            for (j, setting) in settings.iter().enumerate() {
                let pos = [pos[0] + 0.45, pos[1] - 0.5 + ((j + 1) as f32 / (settings.len() + 1) as f32)];
                let size = 0.1 * zoom;
                let mut text = setting.clone();
                text.push_str(": ");
                text.push_str(&n.weight.process.borrow().setting(j));
                if let Some(bb) = fonts.bounding_box(font, size, &text) {
                    let min_x = bb.min.x as f32 / w as f32;
                    let min_y = bb.min.y as f32 / h as f32;
                    let max_x = bb.max.x as f32 / w as f32;
                    let max_y = bb.max.y as f32 / h as f32;
                    let min = inverse_transform(cam, [min_x, min_y]);
                    let max = inverse_transform(cam, [max_x, max_y]);
                    let min = [min[0] * 2., min[1] * 2.];
                    let max = [max[0] * 2., max[1] * 2.];
                    if pos[0] + min[0] < mouse_pos[0] && mouse_pos[0] < pos[0] + max[0] {
                        if pos[1] + min[1] < mouse_pos[1] && mouse_pos[1] < pos[1] + max[1] {
                            return Some(Selection::Setting(i, j));
                        }
                    }
                }
            }
            for s in 0..n.weight.max_out() {
                let pos = output_pos(&n.weight, s, size);
                let pos = [pos[0], -pos[1]];
                if pos[0] - size < mouse_pos[0] && mouse_pos[0] < pos[0] + size {
                    if pos[1] - size < mouse_pos[1] && mouse_pos[1] < pos[1] + size {
                        return Some(Selection::Output(i, s));
                    }
                }
            }
            for t in 0..n.weight.max_in() {
                let pos = input_pos(&n.weight, t, size);
                let pos = [pos[0], -pos[1]];
                if pos[0] - size < mouse_pos[0] && mouse_pos[0] < pos[0] + size {
                    if pos[1] - size < mouse_pos[1] && mouse_pos[1] < pos[1] + size {
                        return Some(Selection::Input(i, t));
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
