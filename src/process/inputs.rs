use std::rc::Rc;
use std::cell::RefCell;

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
    fn modify(&mut self, key: usize, value: String) {
        match key {
            0 => {
                let input = value.split(",").collect::<Vec<_>>();
                self.constant =
                [input[0].trim().parse().unwrap(),
                 input[1].trim().parse().unwrap(),
                 input[2].trim().parse().unwrap(),
                 input[3].trim().parse().unwrap()];
            },
            k => panic!("Unknown option: {}", k),
        }
    }
    fn setting(&self, key: usize) -> String {
        match key {
            0 => format!("{:?}", self.constant),
            k => panic!("Unknown option: {}", k),
        }
    }
    fn settings(&self) -> Vec<String> {
        vec!["color".into()]
    }
    fn max_in(&self) -> u32 {0}
    fn max_out(&self) -> u32 {1}
    fn shader(&self, ctx: &mut Context) -> String {
        let c = self.constant;
        format!("vec4 {} = vec4({}, {}, {}, {});\n", ctx.output(0), c[0], c[1], c[2], c[3])
    }
}
