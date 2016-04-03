use shader::Context;

pub mod inputs;
pub mod combiners;

pub trait Process {
    fn modify(&mut self, key: String, value: String);
    fn options(&self) -> Vec<String>;
    fn max_in(&self) -> u32;
    fn max_out(&self) -> u32;
    fn shader(&self, context: &mut Context) -> String;
}
