use std::rc::Rc;
use std::cell::RefCell;

use vecmath::*;

pub trait Process {
    fn max_in(&self) -> u32;
    fn max_out(&self) -> u32;
    fn process(&self, Vec<[u8; 4]>) -> Vec<[u8; 4]>;
    fn shader(&self, id: usize) -> String;
}

pub struct Constant {
    constant: [u8; 4],
}

impl Constant {
    pub fn new(constant: [u8; 4]) -> Rc<RefCell<Process>> {
        Rc::new(RefCell::new(Constant {
            constant: constant,
        }))
    }
}

impl Process for Constant {
    fn max_in(&self) -> u32 {0}
    fn max_out(&self) -> u32 {1}
    fn process(&self, _: Vec<[u8; 4]>) -> Vec<[u8; 4]> {
        vec![self.constant]
    }
    fn shader(&self, id: usize) -> String {
        let c = self.constant;
        format!("vec4 t{id}_1 = vec4({}, {}, {}, {});\n", c[0] as f32 / 255., c[1] as f32 / 255., c[2] as f32 / 255., c[3] as f32 / 255., id = id)
    }
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
    fn process(&self, input: Vec<[u8; 4]>) -> Vec<[u8; 4]> {
        let i1 = input[0];
        let i2 = input[1];
        let i1 = [i1[0] / 2, i1[1] / 2, i1[2] / 2, i1[3] / 2];
        let i2 = [i2[0] / 2, i2[1] / 2, i2[2] / 2, i2[3] / 2];
        vec![vec4_add(i1, i2)]
    }
    fn shader(&self, id: usize) -> String{
        format!("vec4 t{id}_1 = s{id}_1 * {div} + s{id}_2 * {div};\n", div = 1. / self.max_in() as f32, id = id)
    }
}
