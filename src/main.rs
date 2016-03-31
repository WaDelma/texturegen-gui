#[macro_use]
extern crate glium;
extern crate daggy;
extern crate cam;
extern crate vecmath;

use std::rc::Rc;
use std::cell::RefCell;

use glium::glutin::WindowBuilder;
use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter};
use glium::{DisplayBuild, Blend, Program, Surface};
use glium::index::{IndexBuffer, PrimitiveType};
use glium::draw_parameters::{DrawParameters};
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;
use glium::buffer::Buffer;
use glium::buffer::BufferType::*;
use glium::buffer::BufferMode::*;

use vecmath::*;

use cam::{model_view_projection, Camera, CameraPerspective};

use daggy::{Dag, NodeIndex};

use State::*;
use process::Process;

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

enum State {
    Dragging,
    AddingEdge,
}

fn main() {
    let mut mouse_pos = [0.; 2];
    let mut dag = Dag::<Node, (u32, u32), u32>::new();
    let n1 = dag.add_node(Node::new(process::Constant::new([1.0, 1.0, 1.0, 1.0]), [-2., -2.]));
    let n2 = dag.add_node(Node::new(process::Constant::new([1.0, 1.0, 1.0, 1.0]), [2., -2.]));
    let n3 = dag.add_node(Node::new(process::Blend::new(), [2., 0.]));
    let _ = dag.add_edge(n1, n3, (1, 1)).unwrap();
    let _ = dag.add_edge(n2, n3, (1, 2)).unwrap();
    let n4 = dag.add_node(Node::new(process::Blend::new(), [-2., 0.]));
    let _ = dag.add_edge(n1, n4, (1, 1)).unwrap();
    let n5 = dag.add_node(Node::new(process::Blend::new(), [0., 2.]));
    let _ = dag.add_edge(n3, n5, (1, 2)).unwrap();
    let _ = dag.add_edge(n4, n5, (1, 1)).unwrap();
    let mut selected = None;
    let mut state = None;
    let mut camera = Camera::new([0., 0., 5.]);
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
    let indices = IndexBuffer::new(&display,
                                   PrimitiveType::TrianglesList,
                                   &[0u32, 1, 2, 1, 2, 3]).unwrap();
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
                MouseInput(Pressed, Right) => {
                    dag.add_node(Node::new(process::Constant::new([1.0, 1.0, 1.0, 1.0]), mouse_pos));
                },
                MouseInput(Pressed, Middle) => {
                    state = Some(AddingEdge);
                    selected = find_selected(&dag, mouse_pos);
                },
                MouseInput(Released, Middle) => {
                    if let Some(AddingEdge) = state {
                        if let Some(source) = selected {
                            if let Some(target) = find_selected(&dag, mouse_pos) {
                                let _ = dag.update_edge(NodeIndex::new(source), NodeIndex::new(target), (1, 1));
                            }
                        }
                        state = None;
                        selected = None;
                    }
                },
                MouseInput(Pressed, Left) => {
                    state = Some(Dragging);
                    selected = find_selected(&dag, mouse_pos);
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
                        if let Some(o) = selected {
                            dag.node_weight_mut(NodeIndex::new(o)).unwrap().position = mouse_pos;
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
            let x = node.weight.position[0] - 0.5;
            let y = -node.weight.position[1] - 0.5;
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
        }
        let mut lines = Vec::with_capacity(dag.edge_count());
        if let Some(AddingEdge) = state {
            if let Some(source) = selected {
                let src = dag.node_weight(NodeIndex::new(source)).unwrap();
                let src = [src.position[0], -src.position[1]];
                let trg = [mouse_pos[0], -mouse_pos[1]];
                add_arrow(&mut lines, src, trg, 0.25, 0.10 * TAU);
            }
        }
        for edge in dag.raw_edges() {
            let (s, t) = edge.weight;
            let source = edge.source();
            let target = edge.target();

            let src = dag.node_weight(source).unwrap();
            let src_dis = -0.5 + (s as f32 / (src.max_out() + 1) as f32);
            let src = [src.position[0] + src_dis, -(src.position[1] + 0.5)];

            let trg = dag.node_weight(target).unwrap();
            let trg_dis = -0.5 + (t as f32 / (trg.max_in() + 1) as f32);
            let trg = [trg.position[0] + trg_dis, -(trg.position[1] - 0.5)];

            add_arrow(&mut lines, src, trg, 0.25, 0.10 * TAU);
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

fn find_selected(dag: &Dag<Node, (u32, u32)>, mouse_pos: [f32; 2]) -> Option<usize> {
    dag.raw_nodes()
        .iter()
        .enumerate()
        .filter(|&(_, n)| {
            let pos = n.weight.position;
            if pos[0] - 0.5 < mouse_pos[0] && mouse_pos[0] < pos[0] + 0.5 {
                if pos[1] - 0.5 < mouse_pos[1] && mouse_pos[1] < pos[1] + 0.5 {
                    return true;
                }
            }
            false
        })
        .next()
        .map(|(i, _)| i)
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

mod process {
    use std::rc::Rc;
    use std::cell::RefCell;

    pub trait Process {
        fn max_in(&self) -> u32;
        fn max_out(&self) -> u32;
    }

    pub struct Constant {
        constant: [f64; 4],
    }

    impl Constant {
        pub fn new(constant: [f64; 4]) -> Rc<RefCell<Process>> {
            Rc::new(RefCell::new(Constant {
                constant: constant,
            }))
        }
    }

    impl Process for Constant {
        fn max_in(&self) -> u32 {0}
        fn max_out(&self) -> u32 {1}
    }

    pub struct Blend;

    impl Blend {
        pub fn new() -> Rc<RefCell<Process>> {
            Rc::new(RefCell::new(Blend))
        }
    }

    impl Process for Blend {
        fn max_in(&self) -> u32 {2}
        fn max_out(&self) -> u32 {1}
    }

}
