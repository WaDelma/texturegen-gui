#[macro_use]
extern crate glium;
extern crate daggy;
extern crate cam;
extern crate vecmath;

use std::rc::Rc;
use std::cell::RefCell;

use glium::glutin::WindowBuilder;
use glium::{DisplayBuild, Blend, Program, Surface};
use glium::index::{IndexBuffer, PrimitiveType};
use glium::draw_parameters::{DrawParameters};
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;

use vecmath::*;

use cam::{model_view_projection, Camera, CameraPerspective};

use daggy::{Walker, NodeIndex};

use State::*;
use process::Process;
use dag::PortNumbered;

mod dag;
mod process;

const TAU: f32 = 2. * ::std::f32::consts::PI;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    //pub tex_coords: [f32; 2],
}

implement_vertex!(Vertex, position);//, tex_coords);

struct Node {
    process: Rc<RefCell<Process>>,
    position: [f32; 2],
}

impl Node {
    fn new(process: Rc<RefCell<Process>>, position: [f32; 2],) -> Node {
        Node {
            process: process,
            position: position,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    Node(NodeIndex),
    Input(NodeIndex, u32),
    Output(NodeIndex, u32),
}

fn main() {
    let mut mouse_pos = [0.; 2];
    let mut dag = PortNumbered::<Node, u32>::new();
    let n1 = dag.add_node(Node::new(process::Constant::new([1.0, 1.0, 1.0, 1.0]), [-2., -2.]));
    let n2 = dag.add_node(Node::new(process::Constant::new([1.0, 1.0, 1.0, 1.0]), [2., -2.]));
    let n3 = dag.add_node(Node::new(process::Blend::new(), [2., 0.]));
    let _ = dag.add_edge(n1, 1, n3, 1).unwrap();
    let _ = dag.add_edge(n2, 1, n3, 2).unwrap();
    let n4 = dag.add_node(Node::new(process::Blend::new(), [-2., 0.]));
    let _ = dag.add_edge(n1, 1, n4, 1).unwrap();
    let n5 = dag.add_node(Node::new(process::Blend::new(), [0., 2.]));
    let _ = dag.add_edge(n3, 1, n5, 2).unwrap();
    let _ = dag.add_edge(n4, 1, n5, 1).unwrap();
    let mut selected = None;
    let mut state = None;
    let mut camera = Camera::new([0., 0., -5.]);
    let perspective = CameraPerspective {
        fov: 90f32,
        near_clip: 0.1,
        far_clip: 100.0,
        aspect_ratio: 16. / 9.,
    };


    let display = WindowBuilder::new().build_glium().unwrap();
    let shape = vec![Vertex {
        position: [0., 0.],
    }, Vertex {
        position: [0., 1.],
    }, Vertex {
        position: [1., 0.],
    }, Vertex {
        position: [1., 1.],
    }];
    let vertices = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = IndexBuffer::new(&display, PrimitiveType::TrianglesList, &[0u32, 1, 2, 1, 2, 3]).unwrap();

    let thingy_size = 0.10;
    let thingy_scale = [[thingy_size, 0.0, 0.0, 0.0],
                 [0.0, thingy_size, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]];

    let vertex = r#"
        #version 140
        in vec2 position;
        uniform mat4 matrix;
        void main() {
            gl_Position = matrix * vec4(position, 0.0, 1.0);
        }
    "#;

    let fragment = r#"
        #version 140
        out vec4 color;
        void main() {
            color = vec4(1.0, 1.0, 1.0, 1.0);
        }
    "#;

    let program = Program::from_source(&display, vertex, fragment, None).unwrap();

    let mut running = true;
    while running {
        for event in display.poll_events() {
            use glium::glutin::Event::*;
            use glium::glutin::ElementState::*;
            use glium::glutin::MouseButton::*;
            match event {
                Closed => running = false,
                MouseInput(Pressed, Middle) => {
                    dag.add_node(Node::new(process::Constant::new([1.0, 1.0, 1.0, 1.0]), mouse_pos));
                },
                MouseInput(Pressed, Right) => {
                    state = Some(AddingEdge);
                    match find_selected(&dag, mouse_pos, thingy_size) {
                        s @ Some(Selection::Output(..)) => {
                            selected = s;
                        },
                        Some(Selection::Input(n, i)) => {
                            let (s, i) = dag.remove_edge_to_port(n, i).unwrap();
                            selected = Some(Selection::Output(s, i));
                        },
                        _ => {}
                    }
                },
                MouseInput(Released, Right) => {
                    if let Some(AddingEdge) = state {
                        if let Some(Selection::Output(source, i)) = selected {
                            if let Some(Selection::Input(target, o)) = find_selected(&dag, mouse_pos, thingy_size) {
                                let _ = dag.add_edge(source, i, target, o);
                            }
                        }
                        state = None;
                        selected = None;
                    }
                },
                MouseInput(Pressed, Left) => {
                    state = Some(Dragging);
                    selected = find_selected(&dag, mouse_pos, thingy_size);
                },
                MouseInput(Released, Left) => {
                    if let Some(Dragging) = state {
                        state = None;
                        selected = None;
                    }
                },
                MouseMoved((x, y)) => {
                    let (width, height) = display.get_framebuffer_dimensions();
                    let rel_x = x as f32 / width as f32;
                    let rel_y = y as f32 / height as f32;
                    mouse_pos = line_intersects_plane(&camera, &perspective, rel_x, rel_y);
                    if let Some(Dragging) = state {
                        if let Some(Selection::Node(o)) = selected {
                            dag.node_weight_mut(o).unwrap().position = mouse_pos;
                        }
                    }
                }
                _ => {},
            }
        }
        camera.position = [0., 0., 5.0];
        let view = camera.orthogonal();
        camera.position = [0., 0., -5.0];
        let projection = perspective.projection();
        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 1.0);
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
        for node in dag.raw_nodes() {
            let node = &node.weight;
            let x = node.position[0] - 0.5;
            let y = -node.position[1] - 0.5;
            let mut matrix = [[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [  x,   y, 0.0, 1.0]];
            matrix = model_view_projection(matrix, view, projection);
            let uniforms = uniform! {
                matrix: matrix,
            };

            target.draw(&vertices,
                        &indices,
                        &program,
                        &uniforms,
                        &draw_params)
                  .unwrap();
            for s in 0..node.max_out() {
                let pos = output_pos(node, s + 1, thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let mut matrix = [[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [  x,   y, 0.0, 1.0]];
                matrix = col_mat4_mul(matrix, thingy_scale);
                matrix = model_view_projection(matrix, view, projection);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                target.draw(&vertices,
                            &indices,
                            &program,
                            &uniforms,
                            &draw_params)
                      .unwrap();
            }

            for t in 0..node.max_in() {
                let pos = input_pos(node, t + 1, thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let mut matrix = [[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [  x,   y, 0.0, 1.0]];
                matrix = col_mat4_mul(matrix, thingy_scale);
                matrix = model_view_projection(matrix, view, projection);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                target.draw(&vertices,
                            &indices,
                            &program,
                            &uniforms,
                            &draw_params)
                      .unwrap();
            }
        }
        let mut lines = Vec::with_capacity(dag.edge_count());
        if let Some(AddingEdge) = state {
            if let Some(Selection::Output(source, s)) = selected {
                let src = dag.node_weight(source).unwrap();
                let src = output_pos(src, s, thingy_size);
                let trg = [mouse_pos[0], -mouse_pos[1]];
                add_arrow(&mut lines, src, trg, 0.10, 0.10 * TAU);
            }
        }
        for (source, s, target, t) in dag.edges() {
            let src = dag.node_weight(source).unwrap();
            let trg = dag.node_weight(target).unwrap();
            let src = output_pos(src, s, thingy_size);
            let trg = input_pos(trg, t, thingy_size);
            let trg = [trg[0], trg[1] + thingy_size];
            add_arrow(&mut lines, src, trg, 0.10, 0.10 * TAU);
        }
        let vertices = glium::VertexBuffer::new(&display, &lines).unwrap();
        let indices = (0..lines.len() as u32).collect::<Vec<_>>();
        let indices = IndexBuffer::new(&display, PrimitiveType::LinesList, &indices).unwrap();
        let mut matrix = [[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]];
        matrix = model_view_projection(matrix, view, projection);
        let uniforms = uniform! {
            matrix: matrix,
        };
        target.draw(&vertices,
                    &indices,
                    &program,
                    &uniforms,
                    &draw_params)
              .unwrap();
        target.finish().unwrap();
    }
}

fn input_pos(node: &Node, index: u32, _size: f32) -> [f32; 2] {
    [node.position[0] - 0.5 + (index as f32 / (node.max_in() + 1) as f32),
    -(node.position[1] - 0.5)]
}

fn output_pos(node: &Node, index: u32, size: f32) -> [f32; 2] {
    [node.position[0] - 0.5 + (index as f32 / (node.max_out() + 1) as f32),
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

fn find_selected(dag: &PortNumbered<Node>, mouse_pos: [f32; 2], size: f32) -> Option<Selection> {
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
            for s in 0..n.weight.max_out() {
                let pos = output_pos(&n.weight, s + 1, size);
                let pos = [pos[0], -pos[1]];
                if pos[0] - size < mouse_pos[0] && mouse_pos[0] < pos[0] + size {
                    if pos[1] - size < mouse_pos[1] && mouse_pos[1] < pos[1] + size {
                        return Some(Selection::Output(i, s + 1));
                    }
                }
            }
            for t in 0..n.weight.max_in() {
                let pos = input_pos(&n.weight, t + 1, size);
                let pos = [pos[0], -pos[1]];
                if pos[0] - size < mouse_pos[0] && mouse_pos[0] < pos[0] + size {
                    if pos[1] - size < mouse_pos[1] && mouse_pos[1] < pos[1] + size {
                        return Some(Selection::Input(i, t + 1));
                    }
                }
            }
            None
        })
        .next()
}

pub fn line_intersects_plane(camera: &Camera,
                             pers: &CameraPerspective,
                             rel_x: f32,
                             rel_y: f32) -> [f32; 2] {
    let fov_h = pers.fov.to_radians();
    let fov_v = 2. * f32::atan(f32::tan(fov_h / 2.) * pers.aspect_ratio);

    let near_dist = pers.near_clip;

    let near_width = 2. * near_dist * f32::tan(fov_v / 2.);
    let near_height = 2. * near_dist * f32::tan(fov_h / 2.);

    let forward = camera.forward.clone();
    let up = camera.up.clone();
    let right = camera.right.clone();

    let near_x = near_width * (rel_x - 0.5);
    let near_y = near_height * (rel_y - 0.5);

    let near_x_coord = vec3_scale(right, near_x as f32);
    let near_y_coord = vec3_scale(up, near_y as f32);
    let near_z_coord = vec3_scale(forward, near_dist as f32);

    let ray_dir = vec3_normalized(vec3_add(vec3_add(near_x_coord, near_y_coord), near_z_coord));
    let ray_orig = camera.position.clone();

    let plane_normal = [0., 0., -1.];
    let plane_point = [0., 0., 0.];

    let n_dot_e = vec3_dot(ray_dir, plane_normal);

    if n_dot_e == 0. {
        panic!("Something unexpected happened with targetting. The camera is probably looking \
                away from game plane.");
    }

    let distance = vec3_dot(plane_normal, vec3_sub(plane_point, ray_orig)) / n_dot_e;

    if distance >= 0. {
        let res = vec3_add(ray_orig, vec3_scale(ray_dir, distance));
        [res[0], res[1]]
    } else {
        panic!("Something unexpected happened with targetting.");
    }
}
