use vecmath::*;

pub fn from_window_to_screen((w, h): (u32, u32), pos: [i32; 2]) -> [f32; 2] {
    [pos[0] as f32 / w as f32,
     pos[1] as f32 / h as f32]
}

pub fn from_screen_to_world(cam: [[f32; 4]; 4], pos: [f32; 2]) -> [f32; 2] {
    // let pos = [pos[0] - 0.5, pos[1] - 0.5];
    let pos = inverse_transform(cam, pos);
    [pos[0] * 2., pos[1] * 2.]
}

pub fn from_world_to_screen(cam: [[f32; 4]; 4], pos: [f32; 2]) -> [f32; 2] {
    let pos = transform(cam, pos);
    [pos[0] + 1., -(pos[1] + 1.)]
}

pub fn transform(matrix: [[f32; 4]; 4], vector: [f32; 2]) -> [f32; 2] {
    let m = col_mat4_transform(matrix, [vector[0], vector[1], 0., 1.]);
    [m[0], m[1]]
}

pub fn inverse_transform(matrix: [[f32; 4]; 4], vector: [f32; 2]) -> [f32; 2] {
    let m = col_mat4_transform(mat4_inv(matrix), [vector[0], vector[1], 0., 1.]);
    [m[0], m[1]]
}

pub fn scale(x: f32, y: f32) -> [[f32; 4]; 4] {
    [[x , 0., 0., 0.],
     [0., y , 0., 0.],
     [0., 0., 1., 0.],
     [0., 0., 0., 1.]]
}

pub fn translation(x: f32, y: f32) -> [[f32; 4]; 4] {
    [[1., 0., 0., 0.],
     [0., 1., 0., 0.],
     [0., 0., 1., 0.],
     [x , y , 0., 1.]]
}
