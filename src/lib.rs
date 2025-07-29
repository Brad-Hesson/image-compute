#![allow(dead_code)]
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device, util::align_to};

use crate::{
    buffers::StorageBuffer,
    shaders::{arity1, arity2, plane_fit, unary},
};

mod buffers;
mod shaders;

pub struct Image {
    pub size: [u32; 2],
    size_buffer: StorageBuffer<u32>,
    data_buffer: StorageBuffer<f64>,
    bind_group: arity2::bind_groups::BindGroup0,
}
impl Image {
    pub fn new(
        device: &Device,
        label: Option<&str>,
        size: [u32; 2],
        init_fn: impl FnOnce(&mut [f64]),
    ) -> Self {
        let size_buffer_label = label.map(|name| format!("{name}_size_buffer"));
        let size_buffer = buffers::StorageBuffer::new(
            &device,
            size_buffer_label.as_deref(),
            BufferUsages::UNIFORM,
            2,
            |buf| buf.copy_from_slice(&size),
        );
        let data_buffer_label = label.map(|name| format!("{name}_data_buffer"));
        let data_buffer = buffers::StorageBuffer::new(
            &device,
            data_buffer_label.as_deref(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            size.iter().map(|v| *v as usize).product(),
            init_fn,
        );
        let bind_group = arity2::bind_groups::BindGroup0::from_bindings(
            &device,
            arity2::bind_groups::BindGroupLayout0 {
                size_out: size_buffer.inner.as_entire_buffer_binding(),
                data_out: data_buffer.inner.as_entire_buffer_binding(),
            },
        );
        Self {
            size_buffer,
            data_buffer,
            bind_group,
            size,
        }
    }
    pub fn set_arg_out(&self, pass: &mut ComputePass) {
        self.bind_group.set(pass);
    }
    pub fn set_arg_a(&self, pass: &mut ComputePass) {
        unsafe { std::mem::transmute::<_, &arity2::bind_groups::BindGroup1>(&self.bind_group) }
            .set(pass);
    }
    fn set_arg_b(&self, pass: &mut ComputePass) {
        unsafe { std::mem::transmute::<_, &arity2::bind_groups::BindGroup2>(&self.bind_group) }
            .set(pass);
    }
}

struct Transformer {
    add_pipeline: ComputePipeline,
    subtract_pipeline: ComputePipeline,
    multiply_pipeline: ComputePipeline,
    divide_pipeline: ComputePipeline,
    copy_pipeline: ComputePipeline,
    sum_pipeline: ComputePipeline,
    broadcast_pipeline: ComputePipeline,
    xs_like_pipeline: ComputePipeline,
    ys_like_pipeline: ComputePipeline,
    ones_like_pipeline: ComputePipeline,
}
impl Transformer {
    pub fn new(device: &Device) -> Self {
        Self {
            add_pipeline: arity2::compute::create_add_pipeline(device),
            subtract_pipeline: arity2::compute::create_subtract_pipeline(device),
            multiply_pipeline: arity2::compute::create_multiply_pipeline(device),
            divide_pipeline: arity2::compute::create_divide_pipeline(device),
            copy_pipeline: arity1::compute::create_copy_pipeline(device),
            sum_pipeline: unary::compute::create_sum_pipeline(device),
            broadcast_pipeline: unary::compute::create_broadcast_pipeline(device),
            xs_like_pipeline: arity1::compute::create_xs_like_pipeline(device),
            ys_like_pipeline: arity1::compute::create_ys_like_pipeline(device),
            ones_like_pipeline: arity1::compute::create_ones_like_pipeline(device),
        }
    }
    pub fn add(
        &self,
        pass: &mut ComputePass,
        image_a: &Image,
        image_b: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("add");
        self.simple_pipeline(
            pass,
            &self.add_pipeline,
            arity2::compute::ADD_WORKGROUP_SIZE,
            [image_out, image_a, image_b],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn subtract(
        &self,
        pass: &mut ComputePass,
        image_a: &Image,
        image_b: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("subtract");
        self.simple_pipeline(
            pass,
            &self.subtract_pipeline,
            arity2::compute::SUBTRACT_WORKGROUP_SIZE,
            [image_out, image_a, image_b],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn multiply(
        &self,
        pass: &mut ComputePass,
        image_a: &Image,
        image_b: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("multiply");
        self.simple_pipeline(
            pass,
            &self.multiply_pipeline,
            arity2::compute::MULTIPLY_WORKGROUP_SIZE,
            [image_out, image_a, image_b],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn divide(
        &self,
        pass: &mut ComputePass,
        image_a: &Image,
        image_b: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("divide");
        self.simple_pipeline(
            pass,
            &self.divide_pipeline,
            arity2::compute::DIVIDE_WORKGROUP_SIZE,
            [image_out, image_a, image_b],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn copy(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        out: &Image,
    ) -> Result<(), TransformError> {
        if std::ptr::addr_eq(image, out) {
            return Ok(());
        }
        pass.push_debug_group("copy");
        self.simple_pipeline(
            pass,
            &self.copy_pipeline,
            arity1::compute::COPY_WORKGROUP_SIZE,
            [out, image],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn sum(&self, pass: &mut ComputePass, image: &Image) {
        pass.push_debug_group("sum");
        pass.set_pipeline(&self.sum_pipeline);
        image.set_arg_out(pass);
        image.set_arg_a(pass);
        image.set_arg_b(pass);
        let wg_size = unary::compute::SUM_WORKGROUP_SIZE[0];
        let mut remaining_data = image.size.iter().product::<u32>();
        while remaining_data > 1 {
            let num_workgroups = align_to(remaining_data, wg_size) / wg_size;
            pass.dispatch_workgroups(num_workgroups, 1, 1);
            remaining_data = num_workgroups;
        }
        pass.pop_debug_group();
    }
    pub fn broadcast(&self, pass: &mut ComputePass, image: &Image) -> Result<(), TransformError> {
        pass.push_debug_group("broadcast");
        self.simple_pipeline(
            pass,
            &self.broadcast_pipeline,
            unary::compute::BROADCAST_WORKGROUP_SIZE,
            [image],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn xs_like(
        &self,
        pass: &mut ComputePass,
        image_like: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("xs_like");
        self.simple_pipeline(
            pass,
            &self.xs_like_pipeline,
            arity1::compute::XS_LIKE_WORKGROUP_SIZE,
            [image_out, image_like],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn ys_like(
        &self,
        pass: &mut ComputePass,
        image_like: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("ys_like");
        self.simple_pipeline(
            pass,
            &self.ys_like_pipeline,
            arity1::compute::XS_LIKE_WORKGROUP_SIZE,
            [image_out, image_like],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn ones_like(
        &self,
        pass: &mut ComputePass,
        image_like: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("ones_like");
        self.simple_pipeline(
            pass,
            &self.ones_like_pipeline,
            arity1::compute::ONES_LIKE_WORKGROUP_SIZE,
            [image_out, image_like],
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    #[inline(always)]
    pub fn simple_pipeline<const N: usize>(
        &self,
        pass: &mut ComputePass,
        pipeline: &ComputePipeline,
        wg_size: [u32; 3],
        images: [&Image; N],
    ) -> Result<(), TransformError> {
        let size = check_sizes(images)?;
        pass.set_pipeline(pipeline);
        images.get(0).map(|i| i.set_arg_out(pass));
        images.get(1).map(|i| i.set_arg_a(pass));
        images.get(2).map(|i| i.set_arg_b(pass));
        pass.dispatch_workgroups(
            align_to(size[0], wg_size[0]) / wg_size[0],
            align_to(size[1], wg_size[1]) / wg_size[1],
            1,
        );
        Ok(())
    }
    pub fn sum_ratio(
        &self,
        pass: &mut ComputePass,
        top: &Image,
        bottom: &Image,
        out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("sum_ratio");
        self.copy(pass, bottom, out)?;
        self.sum(pass, out);
        self.broadcast(pass, out)?;
        self.divide(pass, top, out, out)?;
        self.sum(pass, out);
        pass.pop_debug_group();
        Ok(())
    }
    pub fn compute_average(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("compute_average");
        self.ones_like(pass, image, out)?;
        self.sum_ratio(pass, image, out, out)?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn subtract_average(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        average_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("subtract_average");
        self.compute_average(pass, image, average_out)?;
        self.broadcast(pass, &average_out)?;
        self.subtract(pass, &image, &average_out, &image)?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn compute_x_slope(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        slope_out: &Image,
        scratch: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("compute_x_slope");
        self.xs_like(pass, &image, &slope_out)?; // S = x
        self.subtract_average(pass, &slope_out, &scratch)?;
        self.multiply(pass, &slope_out, &slope_out, &scratch)?; // S2 = x^2
        self.sum(pass, &scratch); // S2 = sum(x^2)
        self.multiply(pass, &slope_out, &image, &slope_out)?; // S = x*z
        self.sum(pass, &slope_out); // S = sum(x*z)
        self.divide(pass, &slope_out, &scratch, &slope_out)?; // S = A
        pass.pop_debug_group();
        Ok(())
    }
    pub fn compute_y_slope(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        slope_out: &Image,
        scratch: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("compute_y_slope");
        self.ys_like(pass, &image, &slope_out)?; // S = y
        self.subtract_average(pass, &slope_out, &scratch)?;
        self.multiply(pass, &slope_out, &slope_out, &scratch)?; // S2 = y^2
        self.sum(pass, &scratch); // S2 = sum(y^2)
        self.multiply(pass, &slope_out, &image, &slope_out)?; // S = y*z
        self.sum(pass, &slope_out); // S = sum(y*z)
        self.divide(pass, &slope_out, &scratch, &slope_out)?; // S = A
        pass.pop_debug_group();
        Ok(())
    }
}

pub struct PlaneFitter {
    first: ComputePipeline,
    second_fourth: ComputePipeline,
    third: ComputePipeline,
    third_point_five: ComputePipeline,
    fifth: ComputePipeline,
    buffers: [StorageBuffer<f64>; 8],
    bg: plane_fit::bind_groups::BindGroup2,
    size: [u32; 2],
}
impl PlaneFitter {
    pub fn new(device: &Device, size: [u32; 2]) -> Self {
        let buffers = (0..)
            .map(|i| {
                StorageBuffer::new(
                    device,
                    Some(&format!("scratch_buffer_{i}")),
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    size[0] as usize * size[1] as usize,
                    |_| {},
                )
            })
            .take(8)
            .collect::<Vec<_>>();
        Self {
            first: plane_fit::compute::create_first_pipeline(device),
            second_fourth: plane_fit::compute::create_second_fourth_pipeline(device),
            third: plane_fit::compute::create_third_pipeline(device),
            third_point_five: plane_fit::compute::create_third_point_five_pipeline(device),
            fifth: plane_fit::compute::create_fifth_pipeline(device),
            bg: plane_fit::bind_groups::BindGroup2::from_bindings(
                &device,
                plane_fit::bind_groups::BindGroupLayout2 {
                    xs: buffers[0].inner.as_entire_buffer_binding(),
                    ys: buffers[1].inner.as_entire_buffer_binding(),
                    image_sum__xzs: buffers[2].inner.as_entire_buffer_binding(),
                    ones_sum__x2s: buffers[3].inner.as_entire_buffer_binding(),
                    xs_sum__yzs: buffers[4].inner.as_entire_buffer_binding(),
                    ys_sum__y2s: buffers[5].inner.as_entire_buffer_binding(),
                    debug: buffers[6].inner.as_entire_buffer_binding(),
                    meta_out: buffers[7].inner.as_entire_buffer_binding(),
                },
            ),
            buffers: buffers.try_into().unwrap(),
            size,
        }
    }
    pub fn run(&self, pass: &mut ComputePass) {
        self.bg.set(pass);
        pass.set_pipeline(&self.first);
        dispatch_linear(pass, self.size, plane_fit::compute::FIRST_WORKGROUP_SIZE);
        pass.set_pipeline(&self.second_fourth);
        dispatch_reduction(
            pass,
            self.size,
            plane_fit::compute::SECOND_FOURTH_WORKGROUP_SIZE,
        );
        pass.set_pipeline(&self.third);
        dispatch_linear(pass, self.size, plane_fit::compute::THIRD_WORKGROUP_SIZE);
        pass.set_pipeline(&self.third_point_five);
        dispatch_linear(
            pass,
            self.size,
            plane_fit::compute::THIRD_POINT_FIVE_WORKGROUP_SIZE,
        );
        pass.set_pipeline(&self.second_fourth);
        dispatch_reduction(
            pass,
            self.size,
            plane_fit::compute::SECOND_FOURTH_WORKGROUP_SIZE,
        );
        pass.set_pipeline(&self.fifth);
        dispatch_linear(pass, self.size, plane_fit::compute::FIFTH_WORKGROUP_SIZE);
    }
}

fn dispatch_linear(pass: &mut ComputePass, size: [u32; 2], wg_size: [u32; 3]) {
    pass.dispatch_workgroups(align_to(size[0] * size[1], wg_size[0]) / wg_size[0], 1, 1);
}

fn dispatch_tiles(pass: &mut ComputePass, size: [u32; 2], wg_size: [u32; 3]) {
    pass.dispatch_workgroups(
        align_to(size[0], wg_size[0]) / wg_size[0],
        align_to(size[1], wg_size[1]) / wg_size[1],
        1,
    );
}

fn dispatch_reduction(pass: &mut ComputePass, size: [u32; 2], wg_size: [u32; 3]) {
    let mut remaining_data = size[0] * size[1];
    while remaining_data > 1 {
        let num_workgroups = align_to(remaining_data, wg_size[0]) / wg_size[0];
        pass.dispatch_workgroups(num_workgroups, 1, 1);
        remaining_data = num_workgroups;
    }
}

fn check_sizes<const N: usize>(images: [&Image; N]) -> Result<[u32; 2], TransformError> {
    let sizes = images.iter().map(|i| i.size).collect::<Vec<_>>();
    for size in &sizes[1..] {
        if *size != sizes[0] {
            return Err(TransformError::SizeMismatch(sizes));
        }
    }
    Ok(sizes[0])
}

#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("Size mismatch between image arguments: {:?}", 0)]
    SizeMismatch(Vec<[u32; 2]>),
}

#[cfg(test)]
mod tests {
    use eyre::{Context, Result, bail};
    use high_precision_clock::SimpleHighPrecisionClock;
    use primes::PrimeSet;
    use tracing::info;
    use tracing_subscriber::EnvFilter;
    use wgpu::{
        Adapter, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassTimestampWrites,
        Device, DeviceDescriptor, FeaturesWGPU, FeaturesWebGPU, Instance, PollType, QueryType,
        Queue, RequestAdapterOptions, wgt::QuerySetDescriptor,
    };

    use super::*;

    #[test]
    fn sum_shader() -> Result<()> {
        let (_instance, _adapter, device, queue) = init().context("Init failed")?;
        let transformer = Transformer::new(&device);
        const WIDTH: usize = 512;
        const HEIGHT: usize = 512;
        const SIZE: [u32; 2] = [WIDTH as _, HEIGHT as _];
        let plane_fitter = PlaneFitter::new(&device, SIZE);
        let y_slope = 10000000.0;
        let x_slope = 1.;
        let init_data = |data: &mut [f64]| {
            data.fill(f64::NAN);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let dat = &mut data[y * WIDTH + x];
                    let (x, y) = (x as f64, y as f64);
                    *dat = (y_slope / WIDTH as f64) * x + (x_slope / HEIGHT as f64) * y;
                }
            }
        };
        let original = Image::new(&device, Some("original_image"), SIZE, init_data);
        let z = Image::new(&device, Some("z_image"), SIZE, |_| {});
        let xs = Image::new(&device, Some("xs_image"), SIZE, |_| {});
        let x2s = Image::new(&device, Some("x2s_image"), SIZE, |_| {});
        let ys = Image::new(&device, Some("ys_image"), SIZE, |_| {});
        let y2s = Image::new(&device, Some("y2s_image"), SIZE, |_| {});
        let avg = Image::new(&device, Some("avg_image"), SIZE, |_| {});
        let slope_x = Image::new(&device, Some("slope_x_image"), SIZE, |_| {});
        let curve_x = Image::new(&device, Some("curve_x_image"), SIZE, |_| {});
        let slope_y = Image::new(&device, Some("slope_y_image"), SIZE, |_| {});
        let curve_y = Image::new(&device, Some("curve_y_image"), SIZE, |_| {});
        let scratch = Image::new(&device, Some("scratch_image"), SIZE, |_| {});
        let query_set_buffer = StorageBuffer::<u64>::new(
            &device,
            Some("query_set_buffer"),
            BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            2,
            |_| {},
        );
        let query_set = device.create_query_set(&QuerySetDescriptor {
            count: 2,
            ty: QueryType::Timestamp,
            label: None,
        });
        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;
        unsafe { device.start_graphics_debugger_capture() };
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("test_name"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Test Compute Pass"),
                timestamp_writes: Some(ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            if false {
                // copy image data
                transformer.copy(&mut pass, &original, &z)?;
                // subtract average
                transformer.subtract_average(&mut pass, &z, &avg)?;
                // prepare bases
                transformer.xs_like(&mut pass, &z, &xs)?;
                transformer.subtract_average(&mut pass, &xs, &scratch)?;
                transformer.multiply(&mut pass, &xs, &xs, &x2s)?;
                transformer.ys_like(&mut pass, &z, &ys)?;
                transformer.subtract_average(&mut pass, &ys, &scratch)?;
                transformer.multiply(&mut pass, &ys, &ys, &y2s)?;
                // calculate slopes
                transformer.multiply(&mut pass, &xs, &z, &scratch)?;
                transformer.sum_ratio(&mut pass, &scratch, &x2s, &slope_x)?;
                transformer.multiply(&mut pass, &ys, &z, &scratch)?;
                transformer.sum_ratio(&mut pass, &scratch, &y2s, &slope_y)?;
                transformer.multiply(&mut pass, &x2s, &z, &scratch)?;
                transformer.sum_ratio(&mut pass, &scratch, &x2s, &curve_x)?;
                transformer.multiply(&mut pass, &y2s, &z, &scratch)?;
                transformer.sum_ratio(&mut pass, &scratch, &y2s, &curve_y)?;
                // subtract planes
                transformer.broadcast(&mut pass, &slope_x)?;
                transformer.multiply(&mut pass, &slope_x, &xs, &scratch)?;
                transformer.subtract(&mut pass, &z, &scratch, &z)?;
                transformer.broadcast(&mut pass, &slope_y)?;
                transformer.multiply(&mut pass, &slope_y, &ys, &scratch)?;
                transformer.subtract(&mut pass, &z, &scratch, &z)?;
                // transformer.multiply(&mut pass, &curve_x, &x2s, &scratch)?;
                // transformer.subtract(&mut pass, &z, &scratch, &z)?;
                // transformer.multiply(&mut pass, &curve_y, &y2s, &scratch)?;
                // transformer.subtract(&mut pass, &z, &scratch, &z)?;
            } else {
                original.set_arg_out(&mut pass);
                z.set_arg_a(&mut pass);
                plane_fitter.run(&mut pass);

                // transformer.copy(&mut pass, &z, &original)?;
                // original.set_arg_out(&mut pass);
                // z.set_arg_a(&mut pass);
                // plane_fitter.run(&mut pass);
            }
        }
        encoder.resolve_query_set(&query_set, 0..2, &query_set_buffer.inner, 0);
        device.poll(PollType::WaitForSubmissionIndex(
            queue.submit([encoder.finish()]),
        ))?;
        unsafe { device.stop_graphics_debugger_capture() };

        let debug_download = plane_fitter.buffers[6].queue_download(&device, &queue, ..);
        let meta_download = plane_fitter.buffers[7].queue_download(&device, &queue, ..);
        let image_download = z.data_buffer.queue_download(&device, &queue, ..);
        let x_download = slope_x.data_buffer.queue_download(&device, &queue, ..);
        let y_download = slope_y.data_buffer.queue_download(&device, &queue, ..);
        let timestamps_download = query_set_buffer.queue_download(&device, &queue, ..);
        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;

        let image = image_download.get().unwrap();
        for y in (0..HEIGHT).step_by(HEIGHT / 10) {
            let row = &image[y * WIDTH..];
            for x in (0..WIDTH).step_by(WIDTH / 10) {
                print!("{:9.3e} ", row[x]);
            }
            println!("");
        }
        println!("Image ^   v Debug");
        let image = debug_download.get().unwrap();
        for y in (0..HEIGHT).step_by(HEIGHT / 10) {
            let row = &image[y * WIDTH..];
            for x in (0..WIDTH).step_by(WIDTH / 10) {
                print!("{:9.3e} ", row[x]);
            }
            println!("");
        }
        let times = timestamps_download.get().unwrap();
        println!("{:?} microseconds", (times[1] - times[0]) as f32 / 1000.);
        println!(
            "x: {}, y: {}",
            meta_download.get().unwrap()[0],
            meta_download.get().unwrap()[1]
        );
        println!(
            "x: {}, y: {}",
            x_download.get().unwrap()[0],
            y_download.get().unwrap()[0]
        );
        println!("Actual:");
        println!("x: {}, y: {}", x_slope, y_slope);

        Ok(())
    }

    #[test]
    fn primes_data_race() -> Result<()> {
        let (_instance, _adapter, device, queue) = init().context("Init failed")?;
        let transformer = Transformer::new(&device);

        const WIDTH: usize = 235;
        const HEIGHT: usize = 234;
        const SIZE: [u32; 2] = [WIDTH as _, HEIGHT as _];
        let primes_int = primes::Sieve::new()
            .iter()
            .take(WIDTH * HEIGHT)
            .collect::<Vec<_>>();
        let init_data = |data: &mut [f64]| {
            primes_int
                .iter()
                .zip(data.iter_mut())
                .for_each(|(input, out)| {
                    *out = *input as f64;
                })
        };

        info!("Final prime: {:?}", primes_int.last());
        let image_a = Image::new(&device, Some("image_a"), SIZE, init_data);
        let image_b = Image::new(&device, Some("image_b"), SIZE, init_data);

        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;

        unsafe { device.start_graphics_debugger_capture() };
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("test_name"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Test Compute Pass"),
                timestamp_writes: None,
            });
            transformer.multiply(&mut pass, &image_a, &image_b, &image_b)?;
        }
        device.poll(PollType::WaitForSubmissionIndex(
            queue.submit([encoder.finish()]),
        ))?;
        unsafe { device.stop_graphics_debugger_capture() };
        let buffer_download = image_b.data_buffer.queue_download(&device, &queue, ..);
        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;
        let mismatches = buffer_download
            .get()
            .unwrap()
            .into_iter()
            .zip(&primes_int)
            .enumerate()
            .filter(|(_, (a, b))| (a.sqrt() - **b as f64).abs() > f64::EPSILON)
            .map(|(i, (a, _))| {
                (
                    i,
                    primes::factors(*a as u64)
                        .into_iter()
                        .map(|v| primes_int.binary_search(&v))
                        .collect::<Vec<_>>(),
                )
            })
            .inspect(|(i, v)| println!("{i}: {v:?}"))
            .collect::<Vec<_>>();
        if !mismatches.is_empty() {
            bail!("Failed with {} mismatches", mismatches.len());
        }
        Ok(())
    }

    fn init() -> Result<(Instance, Adapter, Device, Queue)> {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
        info!("{:?}", wgpu::Instance::enabled_backend_features());
        let instance = wgpu::Instance::default();
        let adapter = smol::block_on(instance.request_adapter(&RequestAdapterOptions::default()))
            .context("Adapter request failed")?;
        let (dev, queue) = smol::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: wgpu::Features {
                features_wgpu: FeaturesWGPU::TIMESTAMP_QUERY_INSIDE_PASSES
                    | FeaturesWGPU::SHADER_F64
                    | FeaturesWGPU::SHADER_INT64,
                features_webgpu: FeaturesWebGPU::FLOAT32_FILTERABLE
                    | FeaturesWebGPU::TIMESTAMP_QUERY,
            },
            ..Default::default()
        }))
        .context("Device request failed")?;
        Ok((instance, adapter, dev, queue))
    }

    #[test]
    fn cpu_based() {
        const WIDTH: usize = 256;
        const HEIGHT: usize = 256;
        let mut data = vec![f32::NAN; WIDTH * HEIGHT].into_boxed_slice();
        // for i in 0..data.len() {
        //     data[i] = random();
        // }
        data[0] = 1.;
        data[1] = 2.;
        data[2] = 3.;
        data[3] = 4.;
        data[4] = 5.;
        data[5] = 6.;
        data[6] = 7.;
        data[7] = 8.;
        data[8] = 9.;
        data[9] = 10.;
        data[WIDTH + 0] = 0.;
        data[WIDTH + 1] = 1.;
        data[WIDTH + 2] = 2.;
        data[WIDTH + 3] = 3.;
        data[WIDTH + 4] = 4.;
        data[WIDTH + 5] = 5.;
        data[WIDTH + 6] = 6.;
        data[WIDTH + 7] = 7.;
        data[WIDTH + 8] = 8.;
        data[WIDTH + 9] = 9.;

        let mut clock = SimpleHighPrecisionClock::new(100_000_000);
        let mut times = vec![];
        let mut data_out: Box<[f32]> = Box::new([]);
        for i in 0..1000 {
            let start = clock.now();
            data_out = std::hint::black_box(inside_loop(&data, WIDTH));
            let end = clock.now();
            times.push(end - start);
            if i == 10 {
                clock.calibrate();
            }
        }
        println!("{:?}", &data_out[..10]);
        println!("{:?}", &data_out[WIDTH..][..10]);
        println!(
            "{:?} microseconds",
            times.iter().sum::<u64>() as f64 / times.len() as f64 / 1000.0
        );
    }

    fn inside_loop(data: &[f32], width: usize) -> Box<[f32]> {
        let mut zs = Vec::from(data).into_boxed_slice();
        let mut xs = Box::new_uninit_slice(zs.len());
        let mut ys = Box::new_uninit_slice(zs.len());
        let mut count = 0.;
        let mut zs_sum = 0.;
        for (i, z) in zs.iter().enumerate() {
            if z.is_nan() {
                xs[i].write(f32::NAN);
                ys[i].write(f32::NAN);
            } else {
                count += 1.;
                zs_sum += z;
                xs[i].write((i % width) as f32);
                ys[i].write((i / width) as f32);
            }
        }
        let (xs, ys) = unsafe { (xs.assume_init(), ys.assume_init()) };
        let xs_avg = sum_nan_aware(&xs) / count;
        let ys_avg = sum_nan_aware(&ys) / count;
        let zs_avg = zs_sum / count;
        zs.iter_mut().for_each(|x| *x -= zs_avg);
        let mut xzs = Box::new_uninit_slice(zs.len());
        let mut x2s = Box::new_uninit_slice(zs.len());
        let mut yzs = Box::new_uninit_slice(zs.len());
        let mut y2s = Box::new_uninit_slice(zs.len());
        for (i, z) in zs.iter().enumerate() {
            let x = (i % width) as f32;
            let y = (i / width) as f32;
            xzs[i].write(z * (x - xs_avg));
            x2s[i].write((x - xs_avg) * (x - xs_avg));
            yzs[i].write(z * (y - ys_avg));
            y2s[i].write((y - ys_avg) * (y - ys_avg));
        }
        let (xzs, x2s) = unsafe { (xzs.assume_init(), x2s.assume_init()) };
        let (yzs, y2s) = unsafe { (yzs.assume_init(), y2s.assume_init()) };
        let x_slope = sum_nan_aware(&xzs) / sum_nan_aware(&x2s);
        let y_slope = sum_nan_aware(&yzs) / sum_nan_aware(&y2s);
        for (i, z) in zs.iter_mut().enumerate() {
            let x = (i % width) as f32;
            let y = (i / width) as f32;
            *z = *z - x_slope * (x - xs_avg) - y_slope * (y - ys_avg);
        }
        zs
    }

    #[inline(always)]
    fn sum_nan_aware(data: &[f32]) -> f32 {
        data.iter().filter(|v| !v.is_nan()).sum()
    }
}
