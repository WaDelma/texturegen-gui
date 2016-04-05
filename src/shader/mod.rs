use std::fmt::{self, Formatter, Display};
use std::collections::HashMap;

use glium::backend::Facade;
use glium::Program;

use bit_set::BitSet;

pub struct Context {
    id: usize,
    inputs: HashMap<u32, Identifier>,
    outputs: HashMap<u32, Identifier>,
}

pub struct Inputs<'a>(::std::collections::hash_map::Iter<'a, u32, Identifier>);

impl<'a> Iterator for Inputs<'a> {
    type Item = (u32, Identifier);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(a, b)| (a.clone(), b.clone()))
    }
}

impl Context {
    pub fn new(id: usize, inputs: BitSet, outputs: u32) -> Context {
        Context {
            id: id,
            inputs: inputs.iter().map(|i| (i as u32, Identifier{id: id, itype: Type::Input, index: i as u32})).collect(),
            outputs: (0..outputs).map(|i| (i as u32, Identifier{id: id, itype: Type::Output, index: i as u32})).collect(),
        }
    }

    pub fn input(&self, index: u32) -> Option<Identifier> {
        self.inputs.get(&index).map(Clone::clone)
    }

    pub fn first_input(&self) -> Identifier {
        self.inputs().next().expect("There wasn't any inputs to take first of").1
    }

    pub fn inputs(&self) -> Inputs {
        Inputs(self.inputs.iter())
    }

    pub fn input_len(&self) -> usize {
        self.inputs.len()
    }

    pub fn output(&self, index: u32) -> Identifier {
        *self.outputs.get(&index).expect(&format!("There wasn't output for index: {}", index))
    }
}

#[derive(Clone, Copy)]
enum Type {
    Input,
    Temporary,
    Output,
}

impl Display for Type {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        use self::Type::*;
        match *self {
            Input => "in",
            Temporary => "tmp",
            Output => "out",
        }.fmt(fmt)
    }
}

#[derive(Clone, Copy)]
pub struct Identifier {
    id: usize,
    itype: Type,
    index: u32,
}

impl Display for Identifier {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "{}_{}_{}", self.itype, self.id, self.index)
    }
}

pub struct Shader {
    vertex_snippets: Vec<String>,
    fragment_snippets: Vec<String>,
}

impl Shader {
    pub fn new() -> Shader {
        Shader {
            vertex_snippets: vec![],
            fragment_snippets: vec![],
        }
    }

    pub fn add_vertex<S: Into<String>>(&mut self, snippet: S) {
        self.vertex_snippets.push(snippet.into());
    }

    pub fn add_fragment<S: Into<String>>(&mut self, snippet: S) {
        self.fragment_snippets.push(snippet.into());
    }

    pub fn build<F: Facade>(self, facade: &F) -> Program {
        let mut vertex = String::new();
        vertex.push_str("#version 140\n");
        vertex.push_str("in vec2 position;\n");
        vertex.push_str("uniform mat4 matrix;\n");
        vertex.push_str("void main() {\n");
        for snippet in self.vertex_snippets {
            vertex.push_str(&snippet);
        }
        vertex.push_str("}");

        let mut fragment = String::new();
        fragment.push_str("#version 140\n");
        fragment.push_str("out vec4 color;\n");
        fragment.push_str("void main() {\n");
        for snippet in self.fragment_snippets {
            fragment.push_str(&snippet);
        }
        fragment.push_str("}");
        println!("{}", fragment);

        Program::from_source(facade, &vertex, &fragment, None).unwrap()
    }
}
