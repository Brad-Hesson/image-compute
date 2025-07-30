@group(0) @binding(0)
var<uniform> image_size: vec2<u32>;
@group(0) @binding(1)
var<storage, read_write> image_in: array<f32>;

@group(1) @binding(0)
var<storage, read_write> image_out: array<f32>;
@group(1) @binding(1)
var<storage, read_write> meta_out: array<f64>;

@group(2) @binding(0)
var<storage, read_write> xz: array<f64>;
@group(2) @binding(1)
var<storage, read_write> yz: array<f64>;
@group(2) @binding(2)
var<storage, read_write> xx: array<f64>;
@group(2) @binding(3)
var<storage, read_write> yy: array<f64>;
@group(2) @binding(4)
var<storage, read_write> xxz: array<f64>;
@group(2) @binding(5)
var<storage, read_write> yyz: array<f64>;
@group(2) @binding(6)
var<storage, read_write> xyz: array<f64>;
@group(2) @binding(7)
var<storage, read_write> xxyy: array<f64>;

const SUM_WORKGROUP_SIZE: u32 = 256;

@compute @workgroup_size(256)
fn first(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    meta_out[i] = f64(image_in[i]);
}

@compute @workgroup_size(SUM_WORKGROUP_SIZE)
fn second(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let read_idx = num_workgroups.x * local_index + workgroup_id.x;
    if read_idx < image_len() {
        xz_sum_wg[local_index] = meta_out[read_idx];
        meta_out[read_idx] = 0.;
    } else {
        xz_sum_wg[local_index] = 0.;
    }
    workgroupBarrier();
    var stride = SUM_WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_index < stride {
            xz_sum_wg[local_index] += xz_sum_wg[local_index + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if local_index == 0u {
        meta_out[workgroup_id.x] = xz_sum_wg[0];
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
fn third(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= image_len() { return; }
    let basis = calc_basis(i);
    xz[i] = basis.x * basis.z;
    yz[i] = basis.y * basis.z;
    xx[i] = basis.x * basis.x;
    yy[i] = basis.y * basis.y;
    xxz[i] = basis.x * basis.x * basis.z;
    yyz[i] = basis.y * basis.y * basis.z;
    xyz[i] = basis.x * basis.y * basis.z;
    xxyy[i] = basis.x * basis.x * basis.y * basis.y;
}

var<workgroup> xz_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> yz_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> xx_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> yy_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> xxz_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> yyz_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> xyz_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
var<workgroup> xxyy_sum_wg: array<f64, SUM_WORKGROUP_SIZE>;
@compute @workgroup_size(SUM_WORKGROUP_SIZE)
fn fourth(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let read_idx = num_workgroups.x * local_index + workgroup_id.x;
    if read_idx < image_len() {
        xz_sum_wg[local_index] = xz[read_idx];
        xz[read_idx] = 0.;
        yz_sum_wg[local_index] = yz[read_idx];
        yz[read_idx] = 0.;
        xx_sum_wg[local_index] = xx[read_idx];
        xx[read_idx] = 0.;
        yy_sum_wg[local_index] = yy[read_idx];
        yy[read_idx] = 0.;
        xxz_sum_wg[local_index] = xxz[read_idx];
        xxz[read_idx] = 0.;
        yyz_sum_wg[local_index] = yyz[read_idx];
        yyz[read_idx] = 0.;
        xyz_sum_wg[local_index] = xyz[read_idx];
        xyz[read_idx] = 0.;
        xxyy_sum_wg[local_index] = xxyy[read_idx];
        xxyy[read_idx] = 0.;
    } else {
        xz_sum_wg[local_index] = 0.;
        yz_sum_wg[local_index] = 0.;
        xx_sum_wg[local_index] = 0.;
        yy_sum_wg[local_index] = 0.;
        xxz_sum_wg[local_index] = 0.;
        yyz_sum_wg[local_index] = 0.;
        xyz_sum_wg[local_index] = 0.;
        xxyy_sum_wg[local_index] = 0.;
    }
    workgroupBarrier();
    var stride = SUM_WORKGROUP_SIZE / 2u;
    while stride > 0u {
        if local_index < stride {
            xz_sum_wg[local_index] += xz_sum_wg[local_index + stride];
            yz_sum_wg[local_index] += yz_sum_wg[local_index + stride];
            xx_sum_wg[local_index] += xx_sum_wg[local_index + stride];
            yy_sum_wg[local_index] += yy_sum_wg[local_index + stride];
            xxz_sum_wg[local_index] += xxz_sum_wg[local_index + stride];
            yyz_sum_wg[local_index] += yyz_sum_wg[local_index + stride];
            xyz_sum_wg[local_index] += xyz_sum_wg[local_index + stride];
            xxyy_sum_wg[local_index] += xxyy_sum_wg[local_index + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }
    if local_index == 0u {
        xz[workgroup_id.x] = xz_sum_wg[0];
        yz[workgroup_id.x] = yz_sum_wg[0];
        xx[workgroup_id.x] = xx_sum_wg[0];
        yy[workgroup_id.x] = yy_sum_wg[0];
        xxz[workgroup_id.x] = xxz_sum_wg[0];
        yyz[workgroup_id.x] = yyz_sum_wg[0];
        xyz[workgroup_id.x] = xyz_sum_wg[0];
        xxyy[workgroup_id.x] = xxyy_sum_wg[0];
    }
    workgroupBarrier();
}

var<workgroup> x_slope: f64;
var<workgroup> y_slope: f64;
var<workgroup> xx_slope: f64;
var<workgroup> yy_slope: f64;
var<workgroup> xy_slope: f64;
@compute @workgroup_size(256)
fn fifth(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    if i >= image_len() { return; }
    let basis = calc_basis(i);
    if local_index == 0u {
        let s_xz = xz[0];
        let s_yz = yz[0];
        let s_xx = xx[0];
        let s_yy = yy[0];
        let s_xxz = xxz[0];
        let s_yyz = yyz[0];
        let s_xyz = xyz[0];
        let s_xxyy = xxyy[0];
        x_slope = s_xz / s_xx;
        y_slope = s_yz / s_yy;
        xx_slope = s_xxz / s_xx;
        yy_slope = s_yyz / s_yy;
        xy_slope = s_xyz / s_xxyy;
    }
    workgroupBarrier();
    let plane = x_slope * basis.x + y_slope * basis.y + xx_slope * basis.x * basis.x + yy_slope * basis.y * basis.y + xy_slope * basis.x * basis.y;
    image_out[i] = f32(basis.z - plane);
    if i == 0u {
        meta_out[1] = x_slope;
        meta_out[2] = y_slope;
        meta_out[3] = xx_slope;
        meta_out[4] = yy_slope;
        meta_out[5] = xy_slope;
    }
}

fn image_len() -> u32 {
    return image_size.x * image_size.y;
}
fn calc_basis(i: u32) -> vec3<f64> {
    let w = image_size.x;
    let h = image_size.y;
    let x = i % w;
    let y = i / w;
    let count = f64(image_size.x * image_size.y);
    return vec3(
        f64(i32(x << 1u) - i32(w) + i32(1u)) / 2 / f64(w),
        f64(i32(y << 1u) - i32(h) + i32(1u)) / 2 / f64(h),
        f64(image_in[i]) - meta_out[0] / count
    );
}