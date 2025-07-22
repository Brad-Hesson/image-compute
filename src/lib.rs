#![allow(dead_code)]
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device, util::align_to};

use crate::{buffers::StorageBuffer, shaders::transform};

mod buffers;
mod shaders;

pub struct Image {
    pub size: [u32; 2],
    size_buffer: StorageBuffer<u32>,
    data_buffer: StorageBuffer<f32>,
    bind_group: transform::bind_groups::BindGroup0,
}
impl Image {
    pub fn new(
        device: &Device,
        label: Option<&str>,
        size: [u32; 2],
        init_fn: impl FnOnce(&mut [f32]),
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
        let bindings = transform::bind_groups::BindGroupLayout0 {
            size_out: size_buffer.inner.as_entire_buffer_binding(),
            data_out: data_buffer.inner.as_entire_buffer_binding(),
        };
        let bind_group = transform::bind_groups::BindGroup0::from_bindings(&device, bindings);
        Self {
            size_buffer,
            data_buffer,
            bind_group,
            size,
        }
    }
    fn set_0(&self, pass: &mut ComputePass) {
        self.bind_group.set(pass);
    }
    fn set_1(&self, pass: &mut ComputePass) {
        unsafe { std::mem::transmute::<_, &transform::bind_groups::BindGroup1>(&self.bind_group) }
            .set(pass);
    }
    fn set_2(&self, pass: &mut ComputePass) {
        unsafe { std::mem::transmute::<_, &transform::bind_groups::BindGroup2>(&self.bind_group) }
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
            add_pipeline: transform::compute::create_add_pipeline(device),
            subtract_pipeline: transform::compute::create_subtract_pipeline(device),
            multiply_pipeline: transform::compute::create_multiply_pipeline(device),
            divide_pipeline: transform::compute::create_divide_pipeline(device),
            copy_pipeline: transform::compute::create_copy_pipeline(device),
            sum_pipeline: transform::compute::create_sum_pipeline(device),
            broadcast_pipeline: transform::compute::create_broadcast_pipeline(device),
            xs_like_pipeline: transform::compute::create_xs_like_pipeline(device),
            ys_like_pipeline: transform::compute::create_ys_like_pipeline(device),
            ones_like_pipeline: transform::compute::create_ones_like_pipeline(device),
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
            transform::compute::ADD_WORKGROUP_SIZE,
            image_a,
            image_b,
            image_out,
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
            transform::compute::SUBTRACT_WORKGROUP_SIZE,
            image_a,
            image_b,
            image_out,
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
            transform::compute::MULTIPLY_WORKGROUP_SIZE,
            image_a,
            image_b,
            image_out,
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
            transform::compute::DIVIDE_WORKGROUP_SIZE,
            image_a,
            image_b,
            image_out,
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    pub fn sum(&self, pass: &mut ComputePass, image: &Image) {
        pass.push_debug_group("sum");
        pass.set_pipeline(&self.sum_pipeline);
        image.set_0(pass);
        image.set_1(pass);
        image.set_2(pass);
        let mut remaining = image.size.iter().product::<u32>();
        let wg_size = transform::compute::SUM_WORKGROUP_SIZE[0];
        loop {
            let num_workgroups = align_to(remaining, wg_size) / wg_size;
            pass.dispatch_workgroups(num_workgroups, 1, 1);
            remaining = num_workgroups;
            if remaining == 1 {
                break;
            }
        }
        pass.pop_debug_group();
    }
    pub fn broadcast(&self, pass: &mut ComputePass, image: &Image) -> Result<(), TransformError> {
        pass.push_debug_group("broadcast");
        self.simple_pipeline(
            pass,
            &self.broadcast_pipeline,
            transform::compute::BROADCAST_WORKGROUP_SIZE,
            image,
            image,
            image,
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
            transform::compute::XS_LIKE_WORKGROUP_SIZE,
            image_like,
            image_like,
            image_out,
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
            transform::compute::XS_LIKE_WORKGROUP_SIZE,
            image_like,
            image_like,
            image_out,
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
            transform::compute::ONES_LIKE_WORKGROUP_SIZE,
            image_like,
            image_like,
            image_out,
        )?;
        pass.pop_debug_group();
        Ok(())
    }
    #[inline(always)]
    pub fn simple_pipeline(
        &self,
        pass: &mut ComputePass,
        pipeline: &ComputePipeline,
        wg_size: [u32; 3],
        image_a: &Image,
        image_b: &Image,
        image_out: &Image,
    ) -> Result<(), TransformError> {
        let size = check_sizes([image_a, image_b, image_out])?;
        pass.set_pipeline(pipeline);
        image_out.set_0(pass);
        image_a.set_1(pass);
        image_b.set_2(pass);
        pass.dispatch_workgroups(
            align_to(size[0], wg_size[0]) / wg_size[0],
            align_to(size[1], wg_size[1]) / wg_size[1],
            1,
        );
        Ok(())
    }
    pub fn compute_average(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        average_out: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("compute_average");
        self.ones_like(pass, &image, &average_out)?;
        self.sum(pass, &average_out);
        self.broadcast(pass, &average_out)?;
        self.divide(pass, &image, &average_out, &average_out)?;
        self.sum(pass, &average_out);
        pass.pop_debug_group();
        Ok(())
    }
    pub fn subtract_average(
        &self,
        pass: &mut ComputePass,
        image: &Image,
        scratch: &Image,
    ) -> Result<(), TransformError> {
        pass.push_debug_group("subtract_average");
        self.compute_average(pass, &image, &scratch)?;
        self.broadcast(pass, &scratch)?;
        self.subtract(pass, &image, &scratch, &image)?;
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

        const WIDTH: usize = 1024;
        const HEIGHT: usize = 1024;
        const SIZE: [u32; 2] = [WIDTH as _, HEIGHT as _];
        let init_data = |data: &mut [f32]| {
            data.fill(f32::NAN);
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
        };
        let original = Image::new(&device, Some("original_image"), SIZE, init_data);
        let image = Image::new(&device, Some("working_image"), SIZE, init_data);
        let scratch = Image::new(&device, Some("scratch_image"), SIZE, |_| {});
        let scratch2 = Image::new(&device, Some("scratch2_image"), SIZE, |_| {});
        let scratch3 = Image::new(&device, Some("scratch3_image"), SIZE, |_| {});
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
            // subtract average
            transformer.subtract_average(&mut pass, &image, &scratch)?;
            // subtract x slope
            transformer.compute_x_slope(&mut pass, &original, &scratch, &scratch2)?;
            pass.push_debug_group("subtract x slope");
            transformer.broadcast(&mut pass, &scratch)?; // S = A's
            transformer.xs_like(&mut pass, &original, &scratch2)?; // S2 = x
            transformer.multiply(&mut pass, &scratch, &scratch2, &scratch2)?; // S2 = plane
            transformer.subtract(&mut pass, &image, &scratch2, &image)?;
            pass.pop_debug_group();
            // subtract y slope
            transformer.compute_y_slope(&mut pass, &original, &scratch, &scratch2)?;
            pass.push_debug_group("subtract y slope");
            transformer.broadcast(&mut pass, &scratch)?; // S = A's
            transformer.ys_like(&mut pass, &original, &scratch2)?; // S2 = y
            transformer.multiply(&mut pass, &scratch, &scratch2, &scratch2)?; // S2 = plane
            transformer.subtract(&mut pass, &image, &scratch2, &image)?;
            pass.pop_debug_group();
            transformer.subtract_average(&mut pass, &image, &scratch3)?;
        }
        encoder.resolve_query_set(&query_set, 0..2, &query_set_buffer.inner, 0);
        device.poll(PollType::WaitForSubmissionIndex(
            queue.submit([encoder.finish()]),
        ))?;
        unsafe { device.stop_graphics_debugger_capture() };

        let image_download = image.data_buffer.queue_download(&device, &queue, ..);
        let timestamps_download = query_set_buffer.queue_download(&device, &queue, ..);
        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;

        println!("{:?}", &image_download.get().unwrap()[..100]);
        let times = timestamps_download.get().unwrap();
        println!("{:?} microseconds", (times[1] - times[0]) as f32 / 1000.);
        dbg!(queue.get_timestamp_period());
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
        let init_data = |data: &mut [f32]| {
            primes_int
                .iter()
                .zip(data.iter_mut())
                .for_each(|(input, out)| {
                    *out = *input as f32;
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
            .filter(|(_, (a, b))| (a.sqrt() - **b as f32).abs() > f32::EPSILON)
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
                features_wgpu: FeaturesWGPU::TIMESTAMP_QUERY_INSIDE_PASSES,
                features_webgpu: FeaturesWebGPU::FLOAT32_FILTERABLE
                    | FeaturesWebGPU::TIMESTAMP_QUERY,
            },
            ..Default::default()
        }))
        .context("Device request failed")?;
        Ok((instance, adapter, dev, queue))
    }
}
