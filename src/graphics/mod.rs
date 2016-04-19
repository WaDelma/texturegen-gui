use std::path::PathBuf;
use std::collections::HashMap;

use glium::{VertexBuffer, Program};
use glium::backend::glutin_backend::GlutinFacade;
use glium::index::{IndexBuffer, PrimitiveType};

use self::fonts::Fonts;
use math::*;
use Mat;

pub mod renderer;
pub mod fonts;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
}

pub fn vert(x: f32, y: f32) -> Vertex {
    Vertex {
        position: [x, y],
        tex_coords: [0., 0.],
    }
}

pub fn vertex(x: f32, y: f32, u: f32, v: f32) -> Vertex {
    Vertex {
        position: [x, y],
        tex_coords: [u, v],
    }
}

pub struct Model {
    pub vertices: VertexBuffer<Vertex>,
    pub indices: IndexBuffer<u32>,
}

impl Model {
    pub fn new(vertices: VertexBuffer<Vertex>, indices: IndexBuffer<u32>) -> Model {
        Model {
            vertices: vertices,
            indices: indices,
        }
    }
}

implement_vertex!(Vertex, position, tex_coords);

pub struct RenderContext<'a> {
    pub fonts: Fonts<'a>,
    pub cam: Mat,
    pub programs: HashMap<String, Program>,
    pub models: HashMap<String, Model>,
}

impl<'a> RenderContext<'a> {
    pub fn new(display: &'a GlutinFacade) -> RenderContext<'a> {
        let mut fonts = Fonts::new(display);
        fonts.load("anka",
            PathBuf::from("assets")
                .join("fonts")
                .join("anka")
                .join("bold")
                .with_extension("ttf"));
        
        let mut models = HashMap::new();
        models.insert("node".into(), {
            let (vertices, indices) = (
                [vertex(0., 0., 0., 0.), vertex(0., 1., 0., 1.), vertex(1., 0., 1., 0.), vertex(1., 1., 1., 1.)],
                [0u32, 1, 2, 1, 2, 3]);
            Model::new(VertexBuffer::new(display, &vertices).unwrap(),
            IndexBuffer::new(display, PrimitiveType::TrianglesList, &indices).unwrap())
        });
        models.insert("back".into(), {
            let (vertices, indices) = rounded_rectangle((1., 1.), (0.05, 0.05, 0.05, 0.05));
            Model::new(VertexBuffer::new(display, &vertices).unwrap(),
            IndexBuffer::new(display, PrimitiveType::TrianglesList, &indices).unwrap())
        });

        let mut programs = HashMap::new();
        programs.insert("plain".into(),
            program!(
                display,
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
                }).unwrap());

        RenderContext {
            fonts: fonts,
            cam: matrix([[0.; 4]; 4]),
            models: models,
            programs: programs,
        }
    }
}


fn rounded_rectangle((width, height): (f32, f32), (tlr, trr, blr, brr): (f32, f32, f32, f32)) -> (Vec<Vertex>, Vec<u32>) {
    fn create_corner(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, cur: f32, (x, y): (f32, f32), (a, b, c): (u32, u32, u32), angle: f32) {
        if cur > 0. {
            let num_sides = (0.25 * cur).max(1.);
            for i in 0..num_sides as usize {
                let percent = (i + 1) as f32 / (num_sides + 1.);
                let radians = percent * 0.25 * TAU + angle;
                let (sin, cos) = radians.sin_cos();
                let x = x + sin * cur;
                let y = y - cos * cur;

                vertices.push(vert(x, y));

                let d = vertices.len() as u32 - 1;
                if i == 0  {
                    indices.extend(&[a, b, d][..]);
                } else {
                    indices.extend(&[a, d - 1, d][..]);
                }

                if i == num_sides as usize - 1 {
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

    create_corner(&mut vertices, &mut indices, tlr, (   0. + tlr,     0. + tlr), ( 1,  2, 0), 0.75 * TAU);
    create_corner(&mut vertices, &mut indices, trr, (width - trr,     0. + trr), ( 4,  3, 5), 0.00 * TAU);
    create_corner(&mut vertices, &mut indices, brr, (width - brr, height - brr), (10, 11, 9), 0.25 * TAU);
    create_corner(&mut vertices, &mut indices, blr, (   0. + blr, height - blr), ( 7,  6, 8), 0.50 * TAU);
    (vertices, indices)
}
