use std::rc::Rc;
use std::cell::RefCell;

use shader::Context;

mod inputs;
mod combiners;

pub trait Process {
    fn max_in(&self) -> u32;
    fn max_out(&self) -> u32;
    fn shader(&self, context: &mut Context) -> String;
}
