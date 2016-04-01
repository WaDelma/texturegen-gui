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
