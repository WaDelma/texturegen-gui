use glium::backend::Facade;
use glium::Program;

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
        fragment.push_str("uniform mat4 matrix;\n");
        fragment.push_str("void main() {\n");
        for snippet in self.fragment_snippets {
            fragment.push_str(&snippet);
        }
        fragment.push_str("}");
        // println!("{}", fragment);
        Program::from_source(facade, &vertex, &fragment, None).unwrap()
    }
}
