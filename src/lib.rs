#![allow(dead_code)]
use std::sync::{Arc, OnceLock};

use wgpu::{
    BufferUsages, CommandEncoder, ComputePass, ComputePipeline, Device, QuerySet, QueryType, Queue,
    util::align_to, wgt::QuerySetDescriptor,
};

use crate::{buffers::StorageBuffer, shaders::plane_fit};

mod buffers;
mod shaders;

pub struct Image {
    pub size: [u32; 2],
    size_buffer: StorageBuffer<u32>,
    data_buffer: StorageBuffer<f32>,
    bind_group: plane_fit::bind_groups::BindGroup0,
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
        let bind_group = plane_fit::bind_groups::BindGroup0::from_bindings(
            &device,
            plane_fit::bind_groups::BindGroupLayout0 {
                image_size: size_buffer.inner.as_entire_buffer_binding(),
                image_in: data_buffer.inner.as_entire_buffer_binding(),
            },
        );
        Self {
            size_buffer,
            data_buffer,
            bind_group,
            size,
        }
    }
    pub fn set(&self, pass: &mut ComputePass) {
        self.bind_group.set(pass);
    }
}

#[allow(non_snake_case)]
pub struct PlaneFitterBuffers {
    xz: StorageBuffer<f64>,
    yz: StorageBuffer<f64>,
    bg: plane_fit::bind_groups::BindGroup2,
    size: [u32; 2],
}
impl PlaneFitterBuffers {
    pub fn new(device: &Device, size: [u32; 2]) -> Self {
        let mk_buffer = |s: &'static str| {
            StorageBuffer::new(
                device,
                Some(s),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                size[0] as usize * size[1] as usize,
                |_| {},
            )
        };
        let xz = mk_buffer("xz");
        let yz = mk_buffer("yz");
        Self {
            size,
            bg: plane_fit::bind_groups::BindGroup2::from_bindings(
                &device,
                plane_fit::bind_groups::BindGroupLayout2 {
                    xz: xz.inner.as_entire_buffer_binding(),
                    yz: yz.inner.as_entire_buffer_binding(),
                },
            ),
            xz,
            yz,
        }
    }
}

pub struct PlaneFitter {
    first: ComputePipeline,
    second: ComputePipeline,
    third: ComputePipeline,
    fourth: ComputePipeline,
    fifth: ComputePipeline,
    qs: QuerySet,
    qs_buf: StorageBuffer<u64>,
}
const NUM_TIMESTAMPS: u32 = 10;
impl PlaneFitter {
    pub fn new(device: &Device) -> Self {
        Self {
            first: plane_fit::compute::create_first_pipeline(device),
            second: plane_fit::compute::create_second_pipeline(device),
            third: plane_fit::compute::create_third_pipeline(device),
            fourth: plane_fit::compute::create_fourth_pipeline(device),
            fifth: plane_fit::compute::create_fifth_pipeline(device),
            qs: device.create_query_set(&QuerySetDescriptor {
                label: Some("plane_fitter_qs"),
                ty: QueryType::Timestamp,
                count: NUM_TIMESTAMPS,
            }),
            qs_buf: StorageBuffer::new(
                device,
                Some("plane_fitter_qs_buf"),
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                NUM_TIMESTAMPS as usize,
                |_| {},
            ),
        }
    }
    pub fn run(&self, pass: &mut ComputePass, scratch_buffers: &PlaneFitterBuffers) {
        let mut qs_n = 0;
        let mut wts = |pass: &mut ComputePass| {
            pass.write_timestamp(&self.qs, qs_n);
            qs_n += 1;
        };
        scratch_buffers.bg.set(pass);
        pass.set_pipeline(&self.first);
        wts(pass);
        dispatch_linear(pass, scratch_buffers.size);
        wts(pass);
        pass.set_pipeline(&self.second);
        wts(pass);
        dispatch_reduction(pass, scratch_buffers.size);
        // dispatch_linear(pass, scratch_buffers.size);
        wts(pass);
        pass.set_pipeline(&self.third);
        wts(pass);
        dispatch_linear(pass, scratch_buffers.size);
        wts(pass);
        pass.set_pipeline(&self.fourth);
        wts(pass);
        dispatch_reduction(pass, scratch_buffers.size);
        wts(pass);
        pass.set_pipeline(&self.fifth);
        wts(pass);
        dispatch_linear(pass, scratch_buffers.size);
        wts(pass);
        assert_eq!(NUM_TIMESTAMPS, qs_n);
    }
    pub fn queue_timings_download(
        &self,
        device: &Device,
        queue: &Queue,
    ) -> Arc<OnceLock<[u64; NUM_TIMESTAMPS as usize / 2]>> {
        self.qs_buf.queue_download_with(device, queue, .., |r| {
            [
                r[1] - r[0],
                r[3] - r[2],
                r[5] - r[4],
                r[7] - r[6],
                r[9] - r[8],
            ]
        })
    }
    pub fn resolve_timings(&self, encoder: &mut CommandEncoder) {
        encoder.resolve_query_set(&self.qs, 0..NUM_TIMESTAMPS, &self.qs_buf.inner, 0);
    }
}

fn dispatch_linear(pass: &mut ComputePass, size: [u32; 2]) {
    pass.dispatch_workgroups(
        align_to(size[0] * size[1], plane_fit::WGS) / plane_fit::WGS,
        1,
        1,
    );
}

fn dispatch_reduction(pass: &mut ComputePass, size: [u32; 2]) {
    let mut remaining_data = size[0] * size[1];
    while remaining_data > 1 {
        let num_workgroups = align_to(remaining_data, plane_fit::WGS) / plane_fit::WGS;
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
    use eyre::{Context, Result};
    use tracing::info;
    use tracing_subscriber::EnvFilter;
    use wgpu::{
        Adapter, CommandEncoderDescriptor, ComputePassDescriptor, Device, DeviceDescriptor,
        FeaturesWGPU, FeaturesWebGPU, Instance, PollType, Queue, RequestAdapterOptions,
    };

    use super::*;

    #[test]
    fn output() -> Result<()> {
        let (_instance, _adapter, device, queue) = init().context("Init failed")?;
        const WIDTH: usize = 1024;
        const HEIGHT: usize = 1024;
        const SIZE: [u32; 2] = [WIDTH as _, HEIGHT as _];
        let plane_fitter = PlaneFitter::new(&device);
        let plane_fitter_buffers = PlaneFitterBuffers::new(&device, SIZE);
        let x_slope = 0.0;
        let y_slope = 10.0;
        let offset = 0.0;
        let mut mean = 0.;
        let init_data = |data: &mut [f32]| {
            // data.fill(1.);
            // for i in 0..data.len() {
            //     data[i] = (i / 32) as f32;
            // }
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let dat = &mut data[y * WIDTH + x];
                    let (x, y) = (x as f32 / WIDTH as f32, y as f32 / HEIGHT as f32);
                    // *dat = (x_slope / WIDTH as f32) * x + (y_slope / HEIGHT as f32) * y;
                    let val = x_slope * x + y_slope * y + offset;
                    *dat = val;
                    mean += val as f64;
                }
            }
        };
        let original = Image::new(&device, Some("original_image"), SIZE, init_data);
        mean /= (WIDTH * HEIGHT) as f64;
        let image_out = StorageBuffer::<f32>::new(
            &device,
            None,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            WIDTH * HEIGHT,
            |_| {},
        );
        let meta_out = StorageBuffer::<f64>::new(
            &device,
            None,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            WIDTH * HEIGHT,
            |_| {},
        );
        let out_bg = shaders::plane_fit::bind_groups::BindGroup1::from_bindings(
            &device,
            plane_fit::bind_groups::BindGroupLayout1 {
                image_out: image_out.inner.as_entire_buffer_binding(),
                meta_out: meta_out.inner.as_entire_buffer_binding(),
            },
        );
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
            original.set(&mut pass);
            out_bg.set(&mut pass);
            plane_fitter.run(&mut pass, &plane_fitter_buffers);
        }
        plane_fitter.resolve_timings(&mut encoder);
        device.poll(PollType::WaitForSubmissionIndex(
            queue.submit([encoder.finish()]),
        ))?;
        unsafe { device.stop_graphics_debugger_capture() };
        let meta_download = meta_out.queue_download(&device, &queue, ..16);
        let image_download = image_out.queue_download(&device, &queue, ..);
        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;
        let image = image_download.get().unwrap();
        for y in (0..HEIGHT).step_by(HEIGHT / 10) {
            let row = &image[y * WIDTH..];
            for x in (0..WIDTH).step_by(WIDTH / 10) {
                print!("{:9.3e} ", row[x]);
            }
            println!("");
        }
        println!(
            "a: {}, x: {}, y: {}",
            meta_download.get().unwrap()[0] / SIZE[0] as f64 / SIZE[1] as f64,
            meta_download.get().unwrap()[1],
            meta_download.get().unwrap()[2],
        );
        println!("Actual:");
        println!("a: {}, x: {}, y: {}", mean, x_slope, y_slope);
        Ok(())
    }
    #[test]
    fn timing() -> Result<()> {
        let (_instance, _adapter, device, queue) = init().context("Init failed")?;
        const WIDTH: usize = 1024;
        const HEIGHT: usize = 1024;
        const SIZE: [u32; 2] = [WIDTH as _, HEIGHT as _];
        let plane_fitter = PlaneFitter::new(&device);
        let plane_fitter_buffers = PlaneFitterBuffers::new(&device, SIZE);
        let original = Image::new(&device, Some("original_image"), SIZE, |_| {});
        let image_out = StorageBuffer::<f32>::new(
            &device,
            None,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            WIDTH * HEIGHT,
            |_| {},
        );
        let meta_out = StorageBuffer::<f64>::new(
            &device,
            None,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            WIDTH * HEIGHT,
            |_| {},
        );
        let out_bg = shaders::plane_fit::bind_groups::BindGroup1::from_bindings(
            &device,
            plane_fit::bind_groups::BindGroupLayout1 {
                image_out: image_out.inner.as_entire_buffer_binding(),
                meta_out: meta_out.inner.as_entire_buffer_binding(),
            },
        );
        device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;
        let mut times = vec![0., 0., 0., 0., 0.];
        let mult = queue.get_timestamp_period();
        for i in 1.. {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("test_name"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Test Compute Pass"),
                    timestamp_writes: None,
                });
                original.set(&mut pass);
                out_bg.set(&mut pass);
                plane_fitter.run(&mut pass, &plane_fitter_buffers);
            }
            plane_fitter.resolve_timings(&mut encoder);
            device.poll(PollType::WaitForSubmissionIndex(
                queue.submit([encoder.finish()]),
            ))?;
            let times_download = plane_fitter.queue_timings_download(&device, &queue);
            device.poll(PollType::WaitForSubmissionIndex(queue.submit([])))?;
            let new_times = times_download
                .get()
                .unwrap()
                .iter()
                .map(|v| *v as f64 / 1000. * mult as f64);
            let x = 1. / (i as f64);
            for (mean, new) in times.iter_mut().zip(new_times) {
                *mean = *mean * (1. - x) + new * x;
            }
            if i % 100 == 0 {
                println!(
                    "{x:11.6} {times:9.4?} -> {:9.4} micros",
                    times.iter().sum::<f64>()
                );
            }
        }
        Ok(())
    }
    fn lerp(s: f64, e: f64, t: f64) -> f64 {
        t * e + (1. - t) * s
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
}
