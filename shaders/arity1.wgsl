@group(0) @binding(0)
var<uniform> size_out: vec2<u32>;
@group(0) @binding(1)
var<storage, read_write> data_out: array<f32>;

@group(1) @binding(0)
var<uniform> size_a: vec2<u32>;
@group(1) @binding(1)
var<storage, read_write> data_a: array<f32>;

@compute @workgroup_size(16, 16)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if !all(global_id.xy < size_out) {
        return;
    }
    let pos = global_id.xy;
    write_cell_out(pos, read_cell_a(pos));
}

@compute @workgroup_size(16, 16)
fn ones_like(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if !all(global_id.xy < size_out) {
        return;
    }
    let pos = global_id.xy;
    if is_nan(read_cell_a(pos)) {
        write_cell_out(pos, nan_f32());
    } else {
        write_cell_out(pos, 1.0);
    }
}

@compute @workgroup_size(16, 16)
fn xs_like(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if !all(global_id.xy < size_out) {
        return;
    }
    let pos = global_id.xy;
    if is_nan(read_cell_a(pos)) {
        write_cell_out(pos, nan_f32());
    } else {
        write_cell_out(pos, f32(pos.x));
    }
}

@compute @workgroup_size(16, 16)
fn ys_like(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if !all(global_id.xy < size_out) {
        return;
    }
    let pos = global_id.xy;
    if is_nan(read_cell_a(pos)) {
        write_cell_out(pos, nan_f32());
    } else {
        write_cell_out(pos, f32(pos.y));
    }
}

fn read_cell_a(pos: vec2<u32>) -> f32 {
    return data_a[pos.x + size_a.x * pos.y];
}
fn write_cell_out(pos: vec2<u32>, val: f32) {
    data_out[pos.x + size_out.x * pos.y] = val;
}
fn read_cell_out(pos: vec2<u32>) -> f32 {
    return data_out[pos.x + size_out.x * pos.y];
}

fn is_nan(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    let exponent: u32 = (bits >> 23) & 0xFFu;
    let mantissa: u32 = bits & 0x7FFFFFu;
    return (exponent == 0xFFu) && (mantissa != 0u);
}
fn nan_f32() -> f32 {return bitcast<f32>(0x7FC00000u);}
