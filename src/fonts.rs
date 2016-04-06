use vert;

use std::collections::HashMap;
use std::borrow::Cow;
use std::i32;
use std::cmp;

use glium::{self, Frame, VertexBuffer, Blend, Program, Surface};
use glium::texture::{Texture2d, RawImage2d, MipmapsOption, UncompressedFloatFormat, ClientFormat};
use glium::backend::glutin_backend::GlutinFacade;
use glium::index::{NoIndices, PrimitiveType};
use glium::draw_parameters::DrawParameters;
use glium::uniforms::{MinifySamplerFilter, MagnifySamplerFilter};

use rusttype::{Font, Scale, Point, point, vector, PositionedGlyph};
use rusttype::gpu_cache::Cache;
use rusttype::Rect;

use arrayvec::ArrayVec;

pub struct Fonts<'a> {
    dpi_factor: f32,
    cache: Cache,
    cache_tex: Texture2d,
    program: Program,
    program_bg: Program,
    fonts: HashMap<usize, Font<'a>>,
}

impl<'a> Fonts<'a> {
    pub fn new(display: &GlutinFacade) -> Fonts {
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

    pub fn bounding_box(&self, font: usize, size: f32, text: &str) -> Option<Rect<i32>> {
        let font = self.fonts.get(&font).expect(&format!("Font with id {} didn't exist.", font));
        self.layout(font, Scale::uniform(size * self.dpi_factor), text).1
    }

    fn layout<'b>(&self, font: &'b Font, scale: Scale, text: &str) -> (Vec<PositionedGlyph<'b>>, Option<Rect<i32>>) {
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
        let mut bg = Rect {min: point(i32::MAX, i32::MAX), max: point(i32::MIN, i32::MIN)};
        for glyph in &result {
            if let Some(Rect{min, max}) = glyph.pixel_bounding_box() {
                bg.min.x = cmp::min(bg.min.x, min.x);
                bg.min.y = cmp::min(bg.min.y, min.y);
                bg.max.x = cmp::max(bg.max.x, max.x);
                bg.max.y = cmp::max(bg.max.y, max.y);
            }
        }
        (result, Some(bg))
    }

    pub fn draw_text(&mut self, display: &GlutinFacade, target: &mut Frame, font: usize, size: f32, color: [f32; 4], pos: [f32; 2], text: &str) {
        fn get_rect(min: Point<u32>, tex: &RawImage2d<u8>) -> glium::Rect {
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
