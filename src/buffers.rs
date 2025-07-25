use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::RangeBounds,
    sync::{Arc, OnceLock},
};

use bytemuck::{AnyBitPattern, NoUninit};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, Queue};

pub struct StorageBuffer<T: Clone + NoUninit + AnyBitPattern> {
    pub inner: Buffer,
    pd: PhantomData<T>,
}
impl<T: Clone + NoUninit + AnyBitPattern> StorageBuffer<T> {
    pub fn new(
        device: &Device,
        label: Option<&str>,
        usage: BufferUsages,
        size: usize,
        init_fn: impl FnOnce(&mut [T]),
    ) -> Self {
        let inner = device.create_buffer(&BufferDescriptor {
            label,
            size: (size * std::mem::size_of::<T>()) as u64,
            usage,
            mapped_at_creation: true,
        });
        init_fn(bytemuck::cast_slice_mut(
            inner.get_mapped_range_mut(..).as_mut(),
        ));
        inner.unmap();
        Self {
            inner,
            pd: PhantomData,
        }
    }
    pub fn queue_write(&self, queue: &Queue, offset: usize, data: &[T]) {
        queue.write_buffer(
            &self.inner,
            offset as u64 * size_of::<T>() as u64,
            bytemuck::cast_slice(data),
        );
    }
    pub fn queue_download(
        &self,
        device: &Device,
        queue: &Queue,
        range: impl RangeBounds<BufferAddress>,
    ) -> Arc<OnceLock<Box<[T]>>>
    where
        T: Sync + Send + Debug,
    {
        let buf = Arc::new(std::sync::OnceLock::new());
        let buf_clone = buf.clone();
        wgpu::util::DownloadBuffer::read_buffer(
            device,
            queue,
            &self.inner.slice(range),
            move |db| {
                buf.set(
                    bytemuck::cast_slice(&db.unwrap())
                        .to_vec()
                        .into_boxed_slice(),
                )
                .unwrap();
            },
        );
        buf_clone
    }
}
