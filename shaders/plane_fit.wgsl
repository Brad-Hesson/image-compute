@group(0) @binding(0)
var<uniform> image_in_size: vec2<u32>;
@group(0) @binding(1)
var<storage, read_write> image_in: array<f32>;

@group(1) @binding(0)
var<storage, read_write> image_out: array<f32>;

@group(2) @binding(0)
var<storage, read_write> xs: array<f64>;
@group(2) @binding(1)
var<storage, read_write> ys: array<f64>;
@group(2) @binding(2)
var<storage, read_write> zs: array<f64>;
@group(2) @binding(3)
var<storage, read_write> image_sum__xzs: array<f64>;
@group(2) @binding(4)
var<storage, read_write> ones_sum__x2s: array<f64>;
@group(2) @binding(5)
var<storage, read_write> xs_sum__yzs: array<f64>;
@group(2) @binding(6)
var<storage, read_write> ys_sum__y2s: array<f64>;
@group(2) @binding(7)
var<storage, read_write> meta_out: array<f64>;

const SUM_WORKGROUP_SIZE: u32 = 256;

@compute @workgroup_size(256)
fn first(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    image_sum__xzs[i] = f64(image_in[i]);
    if is_nan_f32(image_in[i]) {
        let nan = nan_f64();
        xs[i] = nan;
        ys[i] = nan;
        xs_sum__yzs[i] = nan;
        ys_sum__y2s[i] = nan;
        ones_sum__x2s[i] = nan;
    } else {
        let x = f64(i % image_width()) / f64(image_in_size.x);
        let y = f64(i / image_width()) / f64(image_in_size.y);
        xs[i] = x;
        ys[i] = y;
        xs_sum__yzs[i] = x;
        ys_sum__y2s[i] = y;
        ones_sum__x2s[i] = 1.0;
    }
}

var<workgroup> image_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> ones_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> xs_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> ys_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
@compute @workgroup_size(SUM_WORKGROUP_SIZE)
fn second_fourth(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let read_idx = num_workgroups.x * local_index + workgroup_id.x;
    if read_idx < image_len() && !is_nan_f64(ones_sum__x2s[read_idx]) {
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
    xs[i] = (xs[i] * count - xs_sum__yzs[0]) / count;
    ys[i] = (ys[i] * count - ys_sum__y2s[0]) / count;
    zs[i] = (f64(image_in[i]) * count - image_sum__xzs[0]) / count;
}

@compute @workgroup_size(256)
fn third_point_five(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    let x = xs[i];
    let y = ys[i];
    let z = zs[i];
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
    let plane = x_slope * xs[i] + y_slope * ys[i];
    image_out[i] = f32(zs[i] - plane);
    if i == 0u {
        meta_out[0] = x_slope;
        meta_out[1] = y_slope;
    }
}

fn image_width() -> u32 {
    return image_in_size.x;
}
fn image_len() -> u32 {
    return image_in_size.x * image_in_size.y;
}
fn is_nan_f32(x: f32) -> bool {
    let bits: u32 = bitcast<u32>(x);
    let exponent: u32 = (bits >> 23) & 0xFFu;
    let mantissa: u32 = bits & 0x7FFFFFu;
    return (exponent == 0xFFu) && (mantissa != 0u);
}
fn is_nan_f64(x: f64) -> bool {
    let bits: u64 = bitcast<u64>(x);
    let exponent: u64 = bits & u64(0x7FF0000000000000);
    let mantissa: u64 = bits & u64(0x000FFFFFFFFFFFFF);
    return (exponent == u64(0x7FF0000000000000)) && (mantissa != u64(0));
}
fn nan_f32() -> f32 {return bitcast<f32>(0x7FC00000u);}
fn nan_f64() -> f64 {return bitcast<f64>(u64(0x7ff8000000000000));}