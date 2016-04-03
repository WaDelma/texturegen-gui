use std::rc::Rc;
use std::cell::RefCell;
use std::any::Any;
use std::collections::HashMap;

use shader::Context;
use process::Process;

pub struct Constant {
    constant: [f32; 4],
}

impl Constant {
    pub fn new(constant: [f32; 4]) -> Rc<RefCell<Process>> {
        Rc::new(RefCell::new(Constant {
            constant: constant,
        }))
    }
}

impl Process for Constant {
    fn modify(&mut self, key: String, value: String) {
        match &*key {
            "color" => {
                let input = value.split("|").collect::<Vec<_>>();
                self.constant =
                [input[0].parse().unwrap(),
                 input[1].parse().unwrap(),
                 input[2].parse().unwrap(),
                 input[3].parse().unwrap()];
            },
            k => panic!("Unknown option: {}", k),
        }
    }
    fn options(&self) -> Vec<String> {
        vec!["color".into()]
    }
    fn max_in(&self) -> u32 {0}
    fn max_out(&self) -> u32 {1}
    fn shader(&self, ctx: &mut Context) -> String {
        let c = self.constant;
        format!("vec4 {} = vec4({}, {}, {}, {});\n", ctx.output(0), c[0], c[1], c[2], c[3])
    }
}
