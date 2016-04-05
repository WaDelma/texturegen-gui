use std::rc::Rc;
use std::cell::RefCell;

use shader::Context;
use process::Process;

pub struct Blend(BlendType, BlendType);

impl Blend {
    pub fn new(color_blend: BlendType, alpha_blend: BlendType) -> Rc<RefCell<Process>> {
        Rc::new(RefCell::new(Blend(color_blend, alpha_blend)))
    }
}

impl Process for Blend {
    fn modify(&mut self, key: usize, value: String) {
        match key {
            0 => self.0 = value.into(),
            1 => self.1 = value.into(),
            k => panic!("Unknown option: {}", k),
        }
    }
    fn setting(&self, key: usize) -> String {
        match key {
            0 => format!("{:?}", self.0),
            1 => format!("{:?}", self.1),
            k => panic!("Unknown option: {}", k),
        }
    }
    fn settings(&self) -> Vec<String> {
        vec!["blend".into(), "alpha".into()]
    }
    fn max_in(&self) -> u32 {2}
    fn max_out(&self) -> u32 {1}
    fn shader(&self, ctx: &mut Context) -> String {
        if ctx.input_len() == 0 {
            return format!("vec4 {} = vec4(0);\n", ctx.output(0));
        }
        if ctx.input_len() == 1 {
            return format!("vec4 {} = {};\n", ctx.output(0), ctx.first_input());
        }
        let mut result = format!("vec4 {} = vec4(", ctx.output(0));
        result.push_str(&self.0.blend(ctx, "rgb"));
        result.push_str(",\n");
        result.push_str(&self.1.blend(ctx, "a"));
        result.push_str(");\n");
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendType {
    Normal,
    Multiply,
    Divide,
    Add,
    Substract,
    Difference,
    Darken,
    Lighten,
    Screen,
    Overlay,
    Hard,
    Soft,
    // Dodge,
    // Burn,
}

impl From<String> for BlendType {
    fn from(input: String) -> BlendType {
        use self::BlendType::*;
        match &*input {
            "normal" => Normal,
            "multiply" => Multiply,
            "divide" => Divide,
            "add" => Add,
            "substract" => Substract,
            "difference" => Difference,
            "darken" => Darken,
            "lighten" => Lighten,
            "screen" => Screen,
            "overlay" => Overlay,
            "hard" => Hard,
            "soft" => Soft,
            i => panic!("Unknown blending type: {}", i),
        }
    }
}

impl BlendType {
    fn blend(&self, ctx: &mut Context, channels: &str) -> String {
        use self::BlendType::*;
        let a = format!("{}.{}", ctx.input(0).unwrap(), channels);
        let b = format!("{}.{}", ctx.input(1).unwrap(), channels);
        let one = format!("one.{}", channels);
        match *self {
            Normal =>     format!("{}", b),
            Multiply =>   format!("{} * {}", a, b),
            Divide =>     format!("{} / {}", a, b),
            Add =>        format!("{} + {}", a, b),
            Substract =>  format!("{} - {}", a, b),
            Difference => format!("abs({} - {})", a, b),
            Darken =>     format!("min({}, {})", a, b),
            Lighten =>    format!("max({}, {})", a, b),
            Screen =>     format!("{one} - ({one} - {}) * ({one} - {})", a, b, one = one),
            Overlay =>    for_each_channel(channels, |c| {
                              let a = format!("{}.{}", ctx.input(0).unwrap(), c);
                              let b = format!("{}.{}", ctx.input(1).unwrap(), c);
                              let one = format!("one.{}", c);
                              format!("{a} < 0.5?\n\
                                    (2 * {a} * {b}):\n\
                                    ({one} - 2 * ({one} - {a}) * ({one} - {b}))", a = a, b = b, one = one)
                          }),
            Hard =>       for_each_channel(channels, |c| {
                              let a = format!("{}.{}", ctx.input(0).unwrap(), c);
                              let b = format!("{}.{}", ctx.input(1).unwrap(), c);
                              let one = format!("one.{}", c);
                              format!("{b} < 0.5?\n\
                              (2 * {a} * {b}):\n\
                              ({one} - 2 * ({one} - {a}) * ({one} - {b}))", a = a, b = b, one = one)
                          }),
            Soft =>       for_each_channel(channels, |c| {
                              let a = format!("{}.{}", ctx.input(0).unwrap(), c);
                              let b = format!("{}.{}", ctx.input(1).unwrap(), c);
                              format!("{b} < 0.5?\n\
                              (2 * {a} * {b} + {a} * {a} - 2 * {a} * {a} * {b}):\n\
                              (2 * sqrt({a}) * {b} - sqrt({a}) + 2 * {a} - 2 * {a} * {b})", a = a, b = b)
                          }),
            // b => panic!("Blending mode \"{:?}\" has not been implemented.", b),
        }
    }
}

fn for_each_channel<F: FnMut(char) -> String>(channels: &str, mut fun: F) -> String {
    let mut result = match channels.len() {
        1 => String::new(),
        n @ 2...4 => format!("vec{}(\n", n),
        n => panic!("Invalid amount of channels: {}", n),
    };
    let mut first = true;
    for c in channels.chars() {
        if !first {
            result.push_str(",\n");
        }
        result.push_str(&fun(c));
        first = false;
    }
    if channels.len() > 1 {
        result.push(')');
    }
    result
}
