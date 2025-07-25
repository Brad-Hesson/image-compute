@group(0) @binding(0)
var<uniform> size_out: vec2<u32>;
@group(0) @binding(1)
var<storage, read_write> data_out: array<f32>;

@group(1) @binding(0)
var<uniform> size_a: vec2<u32>;
@group(1) @binding(1)
var<storage, read_write> data_a: array<f32>;

@group(2) @binding(0)
var<uniform> size_b: vec2<u32>;
@group(2) @binding(1)
var<storage, read_write> data_b: array<f32>;


@compute @workgroup_size(16, 16)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if out_of_bounds(global_id) { return; }
    let pos = global_id.xy;
    write_cell_out(pos, read_cell_a(pos) + read_cell_b(pos));
}

@compute @workgroup_size(16, 16)
fn subtract(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if out_of_bounds(global_id) { return; }
    let pos = global_id.xy;
    write_cell_out(pos, read_cell_a(pos) - read_cell_b(pos));
}

@compute @workgroup_size(16, 16)
fn multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if out_of_bounds(global_id) { return; }
    let pos = global_id.xy;
    write_cell_out(pos, read_cell_a(pos) * read_cell_b(pos));
}

@compute @workgroup_size(16, 16)
fn divide(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if out_of_bounds(global_id) { return; }
    let pos = global_id.xy;
    write_cell_out(pos, read_cell_a(pos) / read_cell_b(pos));
}

fn read_cell_a(pos: vec2<u32>) -> f32 {
    return data_a[pos.x + size_a.x * pos.y];
}
fn read_cell_b(pos: vec2<u32>) -> f32 {
    return data_b[pos.x + size_b.x * pos.y];
}
fn write_cell_out(pos: vec2<u32>, val: f32) {
    data_out[pos.x + size_out.x * pos.y] = val;
}
fn out_of_bounds(global_id: vec3<u32>) -> bool {
    return !all(global_id.xy < size_out);
}