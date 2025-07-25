@group(0) @binding(0)
var<uniform> image_in_size: vec2<u32>;
@group(0) @binding(1)
var<storage, read_write> image_in: array<f32>;

@group(1) @binding(0)
var<uniform> image_out_size: vec2<u32>;
@group(1) @binding(1)
var<storage, read_write> image_out: array<f32>;

@group(2) @binding(0)
var<storage, read_write> xs: array<f32>;
@group(2) @binding(1)
var<storage, read_write> ys: array<f32>;
@group(2) @binding(2)
var<storage, read_write> image_sum__xzs: array<f32>;
@group(2) @binding(3)
var<storage, read_write> ones_sum__x2s: array<f32>;
@group(2) @binding(4)
var<storage, read_write> xs_sum__yzs: array<f32>;
@group(2) @binding(5)
var<storage, read_write> ys_sum__y2s: array<f32>;

const SUM_WORKGROUP_SIZE: u32 = 256;

@compute @workgroup_size(256)
fn first(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    image_sum__xzs[i] = image_in[i];
    if is_nan(image_in[i]) {
        xs[i] = nan_f32();
        ys[i] = nan_f32();
        xs_sum__yzs[i] = nan_f32();
        ys_sum__y2s[i] = nan_f32();
        ones_sum__x2s[i] = nan_f32();
    } else {
        xs[i] = f32(i % image_width());
        ys[i] = f32(i / image_width());
        xs_sum__yzs[i] = f32(i % image_width());
        ys_sum__y2s[i] = f32(i / image_width());
        ones_sum__x2s[i] = 1.0;
    }
}

var<workgroup> image_sum_wg: array<f32, SUM_WORKGROUP_SIZE>;
var<workgroup> ones_sum_wg: array<f32, SUM_WORKGROUP_SIZE>;
var<workgroup> xs_sum_wg: array<f32, SUM_WORKGROUP_SIZE>;
var<workgroup> ys_sum_wg: array<f32, SUM_WORKGROUP_SIZE>;
@compute @workgroup_size(SUM_WORKGROUP_SIZE)
fn second_fourth(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let read_idx = SUM_WORKGROUP_SIZE * workgroup_id.x + local_index;
    if read_idx < image_len() && !is_nan(ones_sum__x2s[read_idx]) {
        image_sum_wg[local_index] = image_sum__xzs[read_idx];
        image_sum__xzs[read_idx] = 0.;
        ones_sum_wg[local_index] = ones_sum__x2s[read_idx];
        ones_sum__x2s[read_idx] = 0.;
        xs_sum_wg[local_index] = xs_sum__yzs[read_idx];
        xs_sum__yzs[read_idx] = 0.;
        ys_sum_wg[local_index] = ys_sum__y2s[read_idx];
        ys_sum__y2s[read_idx] = 0.;
    } else {
        image_sum_wg[local_index] = 0.;
        ones_sum_wg[local_index] = 0.;
        xs_sum_wg[local_index] = 0.;
        ys_sum_wg[local_index] = 0.;
    }
    workgroupBarrier();
    var stride = SUM_WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_index < stride {
            image_sum_wg[local_index] += image_sum_wg[local_index + stride];
            ones_sum_wg[local_index] += ones_sum_wg[local_index + stride];
            xs_sum_wg[local_index] += xs_sum_wg[local_index + stride];
            ys_sum_wg[local_index] += ys_sum_wg[local_index + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if local_index == 0u {
        image_sum__xzs[workgroup_id.x] = image_sum_wg[0];
        ones_sum__x2s[workgroup_id.x] = ones_sum_wg[0];
        xs_sum__yzs[workgroup_id.x] = xs_sum_wg[0];
        ys_sum__y2s[workgroup_id.x] = ys_sum_wg[0];
    }
    workgroupBarrier();
}


@compute @workgroup_size(256)
fn third(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    let count = ones_sum__x2s[0];
    xs[i] = xs[i] - (xs_sum__yzs[0] / count);
    ys[i] = ys[i] - (ys_sum__y2s[0] / count);
    image_out[i] = image_in[i] - (image_sum__xzs[0] / count);
}

@compute @workgroup_size(256)
fn third_point_five(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    let x = xs[i];
    let y = ys[i];
    let z = image_out[i];
    image_sum__xzs[i] = x * z;
    ones_sum__x2s[i] = x * x;
    xs_sum__yzs[i] = y * z;
    ys_sum__y2s[i] = y * y;
}

@compute @workgroup_size(256)
fn fifth(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    let x_slope = image_sum__xzs[0] / ones_sum__x2s[0];
    let y_slope = xs_sum__yzs[0] / ys_sum__y2s[0];
    image_out[i] -= (x_slope * xs[i] + y_slope * ys[i]);
}


fn image_width() -> u32 {
    return image_in_size.x;
}
fn image_len() -> u32 {
    return image_out_size.x * image_out_size.y;
}
fn is_nan(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    let exponent: u32 = (bits >> 23) & 0xFFu;
    let mantissa: u32 = bits & 0x7FFFFFu;
    return (exponent == 0xFFu) && (mantissa != 0u);
}
fn nan_f32() -> f32 {return bitcast<f32>(0x7FC00000u);}