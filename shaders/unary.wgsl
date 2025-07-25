@group(0) @binding(0)
var<uniform> size_out: vec2<u32>;
@group(0) @binding(1)
var<storage, read_write> data_out: array<f32>;

const SUM_WORKGROUP_SIZE: u32 = 256;
var<workgroup> sum_wg_array: array<f32, SUM_WORKGROUP_SIZE>;
@compute @workgroup_size(SUM_WORKGROUP_SIZE)
fn sum(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let data_len = size_out.x * size_out.y;
    let this_threads_data_index = SUM_WORKGROUP_SIZE * workgroup_id.x + local_index;
    if this_threads_data_index < data_len && !is_nan(data_out[this_threads_data_index]) {
        sum_wg_array[local_index] = data_out[this_threads_data_index];
        data_out[this_threads_data_index] = 0.;
    } else {
        sum_wg_array[local_index] = 0.;
    }
    workgroupBarrier();
    var stride = SUM_WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_index < stride {
            sum_wg_array[local_index] += sum_wg_array[local_index + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if local_index == 0u {
        data_out[workgroup_id.x] = sum_wg_array[0];
    }
}

@compute @workgroup_size(16, 16)
fn broadcast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if !all(global_id.xy < size_out) {
        return;
    }
    let pos = global_id.xy;
    write_cell_out(pos, read_cell_out(vec2(0u)));
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
