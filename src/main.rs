#[macro_use]
extern crate glium;
extern crate daggy;
extern crate cam;
extern crate vecmath;
extern crate bit_set;

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

use glium::VertexBuffer;
use glium::backend::Facade;
use glium::glutin::WindowBuilder;
use glium::{DisplayBuild, Blend, Program, Surface};
use glium::index::{IndexBuffer, PrimitiveType};
use glium::draw_parameters::{DrawParameters};
use glium::draw_parameters::LinearBlendingFactor::*;
use glium::draw_parameters::BlendingFunction::*;

use bit_set::BitSet;

use vecmath::*;

use cam::{model_view_projection, Camera, CameraPerspective};

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
}

fn main() {
    let display = WindowBuilder::new().build_glium().unwrap();
    let mut mouse_pos = [0.; 2];
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
    let mut camera = Camera::new([0., 0., -5.]);
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
    line_program.add_vertex("gl_Position = matrix * vec4(position, 0.0, 1.0);\n");
    line_program.add_fragment("color = vec4(1.);\n");
    let line_program = line_program.build(&display);
    let thingy_program = &line_program;
    let back_program = &line_program;

    let mut running = true;
    while running {
        let (width, height) = display.get_framebuffer_dimensions();
        let perspective = CameraPerspective {
            fov: 90.,
            near_clip: 0.1,
            far_clip: 100.0,
            aspect_ratio: width as f32 / height as f32,
        };
        for event in display.poll_events() {
            use glium::glutin::Event::*;
            use glium::glutin::ElementState::*;
            use glium::glutin::MouseButton as Mouse;
            use glium::glutin::VirtualKeyCode as Key;
            match event {
                Closed => running = false,
                ReceivedCharacter(c) => {
                    if let Some(Writing) = state {
                        if !c.is_whitespace() && !c.is_control() {
                            text.push(c);
                            println!("text: {:?}", text);
                        }
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Back)) => {
                    if let Some(Writing) = state {
                        text.pop();
                        println!("text: {:?}", text);
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Escape)) => {
                    if let Some(Writing) = state {
                        println!("Stopped writing!");
                        text.clear();
                        selected = None;
                        state = None;
                    }
                },
                KeyboardInput(Pressed, _, Some(Key::Return)) => {
                    if let Some(Writing) = state {
                        if let Some(Selection::Node(n)) = selected {
                            let mut split = text.split(":");
                            dag.node_weight(n)
                                .unwrap()
                                .process
                                .borrow_mut()
                                .modify(split.next().unwrap().into(), split.next().unwrap().into());
                            update_dag(&display, &dag, n);
                        }
                        text.clear();
                        selected = None;
                        state = None;
                    } else if let None = state {
                        if let n @ Some(Selection::Node(_)) = find_selected(&dag, mouse_pos, thingy_size) {
                            println!("Started writing!");
                            selected = n;
                            state = Some(Writing);
                        }
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
                MouseInput(Pressed, Mouse::Middle) => {
                    if let None = state {
                        if let Some(Selection::Node(n)) = find_selected(&dag, mouse_pos, thingy_size) {
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
                        match find_selected(&dag, mouse_pos, thingy_size) {
                            s @ Some(Selection::Output(..)) => {
                                selected = s;
                            },
                            Some(Selection::Input(n, i)) => {
                                let (s, i) = dag.remove_edge_to_port(n, i).unwrap();
                                update_dag(&display, &dag, n);
                                selected = Some(Selection::Output(s, i));
                            },
                            _ => {}
                        }
                    }
                },
                MouseInput(Released, Mouse::Left) => {
                    if let Some(AddingEdge) = state {
                        if let Some(Selection::Output(source, i)) = selected {
                            if let Some(Selection::Input(target, o)) = find_selected(&dag, mouse_pos, thingy_size) {
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
                        state = Some(Dragging);
                        selected = find_selected(&dag, mouse_pos, thingy_size);
                    }
                },
                MouseInput(Released, Mouse::Right) => {
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
        camera.position = [0., 0., 5.];
        let view = camera.orthogonal();
        camera.position = [0., 0., -5.];
        let projection = perspective.projection();
        let mut target = display.draw();
        target.clear_color(0.0157, 0.0173, 0.0204, 1.);//0., 0., 0., 1.);
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
            //Background of node
            let mut matrix = [[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [x , y , 0., 1.]];
            matrix = model_view_projection(matrix, view, projection);
            let program = back_program;
            let uniforms = uniform! {
                matrix: matrix,
            };

            let vertices = &back_model.0;
            let indices = &back_model.1;
            target.draw(vertices,
                        indices,
                        &program,
                        &uniforms,
                        &draw_params)
                  .unwrap();
            //Node
            let x = x + 0.05;
            let y = y + 0.05;
            let mut matrix = [[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [x , y , 0., 1.]];
            matrix = col_mat4_mul(matrix,
                             [[0.9, 0. , 0., 0.],
                              [0. , 0.9, 0., 0.],
                              [0. , 0. , 1., 0.],
                              [0. , 0. , 0., 1.]]);
            matrix = model_view_projection(matrix, view, projection);
            let program = node.program.borrow();
            let program = program.as_ref().unwrap();
            let uniforms = uniform! {
                matrix: matrix,
            };

            let vertices = &node_model.0;
            let indices = &node_model.1;
            target.draw(vertices,
                        indices,
                        &program,
                        &uniforms,
                        &draw_params)
                  .unwrap();
            for s in 0..node.max_out() {
                let pos = output_pos(node, s, thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let mut matrix = [[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [x , y , 0., 1.]];
                matrix = col_mat4_mul(matrix, thingy_scale);
                matrix = model_view_projection(matrix, view, projection);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                let vertices = &thingy_model.0;
                let indices = &thingy_model.1;
                target.draw(vertices,
                            indices,
                            &thingy_program,
                            &uniforms,
                            &draw_params)
                      .unwrap();
            }

            for t in 0..node.max_in() {
                let pos = input_pos(node, t, thingy_size);
                let x = pos[0] - thingy_size / 2.;
                let y = pos[1];
                let mut matrix = [[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [x , y , 0., 1.]];
                matrix = col_mat4_mul(matrix, thingy_scale);
                matrix = model_view_projection(matrix, view, projection);
                let uniforms = uniform! {
                    matrix: matrix,
                };
                let vertices = &thingy_model.0;
                let indices = &thingy_model.1;
                target.draw(vertices,
                            indices,
                            &thingy_program,
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
        let vertices = glium::VertexBuffer::new(&display, &lines).unwrap();
        let indices = (0..lines.len() as u32).collect::<Vec<_>>();
        let indices = IndexBuffer::new(&display, PrimitiveType::LinesList, &indices).unwrap();
        let mut matrix = [[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]];
        matrix = model_view_projection(matrix, view, projection);
        let uniforms = uniform! {
            matrix: matrix,
        };
        target.draw(&vertices,
                    &indices,
                    &line_program,
                    &uniforms,
                    &draw_params)
              .unwrap();
        target.finish().unwrap();
    }
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
