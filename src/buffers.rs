use std::{marker::PhantomData, ops::RangeBounds};

use bytemuck::AnyBitPattern;
use wgpu::{
    Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferSlice, BufferUsages, BufferView,
    Device, Queue, wgc::device::queue,
};

pub struct StorageBuffer<T: Clone + bytemuck::NoUninit + AnyBitPattern>(pub Buffer, PhantomData<T>);
impl<T: Clone + bytemuck::NoUninit + AnyBitPattern> StorageBuffer<T> {
    pub fn new_with(
        device: &Device,
        label: Option<&str>,
        size: usize,
        fill: T,
        usage: BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: (size * std::mem::size_of::<T>()) as u64,
            usage,
            mapped_at_creation: true,
        });
        bytemuck::cast_slice_mut(buffer.slice(..).get_mapped_range_mut().as_mut()).fill(fill);
        buffer.unmap();
        Self(buffer, PhantomData)
    }
    pub fn new_as(device: &Device, label: Option<&str>, data: &[T], usage: BufferUsages) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: (data.len() * std::mem::size_of::<T>()) as u64,
            usage,
            mapped_at_creation: true,
        });
        bytemuck::cast_slice_mut(buffer.slice(..).get_mapped_range_mut().as_mut())
            .copy_from_slice(data);
        buffer.unmap();
        Self(buffer, PhantomData)
    }
    pub fn new(device: &Device, label: Option<&str>, len: usize, usage: BufferUsages) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: (len * std::mem::size_of::<T>()) as u64,
            usage,
            mapped_at_creation: false,
        });
        Self(buffer, PhantomData)
    }
    pub fn write(&self, queue: &Queue, offset: usize, data: &[T]) {
        queue.write_buffer(
            &self.0,
            offset as u64 * size_of::<T>() as u64,
            bytemuck::cast_slice(data),
        );
    }
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice {
        self.0.slice(bounds)
    }
    pub fn as_entire_buffer_binding(&self) -> BufferBinding {
        self.0.as_entire_buffer_binding()
    }
}
