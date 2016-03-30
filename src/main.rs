#[macro_use]
extern crate glium;
extern crate daggy;
extern crate cam;

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

use cam::{model_view_projection, Camera, CameraPerspective};

use daggy::Dag;

use process::*;

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
}

fn main() {
	let mut dag = Dag::<Node, u32, u32>::new();
    let n1 = dag.add_node(Node::new(Constant::new([1.0, 1.0, 1.0, 1.0]), [0., 0.]));
    let n2 = dag.add_node(Node::new(Constant::new([1.0, 1.0, 1.0, 1.0]), [1., -1.]));
    let e1 = dag.add_edge(n1, n2, 1u32).unwrap();
    let n3 = dag.add_node(Node::new(Constant::new([1.0, 1.0, 1.0, 1.0]), [-1., -1.]));
    let e2 = dag.add_edge(n1, n3, 2u32).unwrap();
    let n4 = dag.add_node(Node::new(Constant::new([1.0, 1.0, 1.0, 1.0]), [0., -2.]));
    let e3 = dag.add_edge(n2, n4, 1u32).unwrap();
    let e4 = dag.add_edge(n3, n4, 1u32).unwrap();
    let camera = Camera::new([0., 0., 5.]);
    let perspective = CameraPerspective {
        fov: 90f32,
        near_clip: 0.1,
        far_clip: 100.0,
        aspect_ratio: 16. / 9.,
    };


    let display = WindowBuilder::new().build_glium().unwrap();
    let shape = vec![Vertex {
        position: [0., 0.],
        //tex_coords: [0.0, 0.0],
    }, Vertex {
        position: [0., 1.],
        //tex_coords: [0.0, 1.0],
    }, Vertex {
        position: [1., 0.],
        //tex_coords: [1.0, 0.0],
    }, Vertex {
        position: [1., 1.],
        //tex_coords: [1.0, 1.0],
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
            match event {
                Closed => running = false,
                /*MouseInput(Pressed, Left) => { TODO: Implement grabbing.
                    selected = Some();
                }
                MouseInput(Released, Left) => {
                    selected = None;
                }*/
                _ => {},
            }
        }
        let view = camera.orthogonal();
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
            let x = node.weight.position[0];
            let y = node.weight.position[1];
            let mut matrix = [[1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [  x,   y, 0.0, 1.0]];
            matrix = model_view_projection(matrix, view, projection);
            let uniforms = uniform! {
                matrix: matrix,
                //position: node.weight.position,
            };
            
            target.draw(&vertices,
                        &indices,
                        &program,
                        &uniforms,
                        &draw_params)
                  .unwrap();
        }
        for edge in dag.raw_edges() {
            
        }
        target.finish().unwrap();
    }
}

mod process {
    use Node;

    use std::rc::Rc;
    use std::cell::RefCell;
    
    pub trait Process {

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

    }
}
