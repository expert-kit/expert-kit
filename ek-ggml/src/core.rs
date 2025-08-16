use std::ptr::NonNull;

use crate::bindings::{ggml_dup_tensor, ggml_n_dims, ggml_tensor_overhead, ggml_type_size};

#[allow(warnings)]
pub(crate) mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug)]
pub struct Context {
    ptr: NonNull<bindings::ggml_context>,
}

impl Context {
    pub fn new(size: usize) -> Self {
        let params = bindings::ggml_init_params {
            mem_buffer: std::ptr::null_mut(),
            mem_size: size,
            no_alloc: false,
        };

        let ctx = unsafe { bindings::ggml_init(params) };

        Self {
            ptr: NonNull::new(ctx).unwrap(),
        }
    }

    pub fn create_tensor(
        &self,
        shape: &[i64],
        kind: Kind,
        data: Vec<u8>,
    ) -> Result<SharedTensor, Error> {
        let ne = shape.iter().rev().cloned().collect::<Vec<_>>();
        let tensor = unsafe {
            bindings::ggml_new_tensor(
                self.ptr.as_ptr(),
                kind as _,
                ne.len() as _,
                ne.as_ptr() as _,
            )
        };

        let tensor = unsafe { &mut *tensor };
        Tensor {
            ctx: self.ptr,
            ptr: tensor,
        }
        .set_data(&data)?;

        Ok(SharedTensor(Tensor {
            ctx: self.ptr,
            ptr: tensor,
        }))
    }

    pub fn create_graph<const N: usize>(
        &self,
        compute: impl FnOnce(&TensorAllocator) -> ([Tensor; N], Tensor),
    ) -> Graph<N> {
        let graph = unsafe { bindings::ggml_new_graph(self.ptr.as_ptr()) };

        let allocator = TensorAllocator { ctx: self.ptr };
        let (inputs, output) = compute(&allocator);

        unsafe { bindings::ggml_build_forward_expand(graph, output.ptr) };

        Graph {
            ctx: NonNull::new(self.ptr.as_ptr()).unwrap(),
            ptr: NonNull::new(graph).unwrap(),
            inputs: inputs.map(|input| input.ptr),
            output: output.ptr,
        }
    }
}

unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { bindings::ggml_free(self.ptr.as_ptr()) };
    }
}

pub struct Graph<const N: usize> {
    ctx: NonNull<bindings::ggml_context>,
    ptr: NonNull<bindings::ggml_cgraph>,
    inputs: [*mut bindings::ggml_tensor; N],
    output: *mut bindings::ggml_tensor,
}

impl<const N: usize> Graph<N> {
    pub fn overhead() -> usize {
        unsafe { bindings::ggml_graph_overhead() }
    }

    pub fn inputs_kind(&self) -> [Kind; N] {
        self.inputs.map(|tensor| {
            let inner = unsafe { &*tensor };
            inner.type_.into()
        })
    }

    pub fn output_kind(&self) -> Kind {
        let inner = unsafe { &*self.output };
        inner.type_.into()
    }

    pub fn inputs_shape(&self) -> [Vec<i64>; N] {
        self.inputs.map(|tensor| {
            let n_dim = unsafe { ggml_n_dims(tensor) };
            let inner = unsafe { &*tensor };
            inner.ne.iter().take(n_dim as _).rev().cloned().collect()
        })
    }

    pub fn output_shape(&self) -> Vec<i64> {
        let n_dim = unsafe { ggml_n_dims(self.output) };
        let inner = unsafe { &*self.output };
        inner.ne.iter().take(n_dim as _).rev().cloned().collect()
    }

    pub fn compute(&mut self, inputs: [&[u8]; N], n_threads: usize) -> Result<Vec<u8>, Error> {
        for (&tensor, &data) in self.inputs.iter().zip(inputs.iter()) {
            let tensor = Tensor {
                ctx: self.ctx,
                ptr: tensor,
            };
            tensor.set_data(data)?;
        }

        unsafe {
            bindings::ggml_graph_compute_with_ctx(
                self.ctx.as_ptr(),
                self.ptr.as_mut(),
                n_threads as _,
            )
        };

        let output = unsafe { &mut *self.output };
        let output = Tensor {
            ctx: self.ctx,
            ptr: output,
        };

        Ok(output.get_data().to_vec())
    }
}

unsafe impl<const N: usize> Send for Graph<N> {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Kind {
    F32 = bindings::ggml_type_GGML_TYPE_F32,
    BF16 = bindings::ggml_type_GGML_TYPE_BF16,
}

impl Kind {
    #[inline]
    pub fn size(&self) -> usize {
        let ggml_type: bindings::ggml_type = *self as _;
        unsafe { bindings::ggml_type_size(ggml_type) }
    }
}

impl From<bindings::ggml_type> for Kind {
    #[inline]
    fn from(ggml_type: bindings::ggml_type) -> Self {
        match ggml_type {
            bindings::ggml_type_GGML_TYPE_F32 => Kind::F32,
            bindings::ggml_type_GGML_TYPE_BF16 => Kind::BF16,
            _ => unreachable!(),
        }
    }
}

pub struct TensorAllocator {
    ctx: NonNull<bindings::ggml_context>,
}

impl TensorAllocator {
    pub fn borrow<'a>(&self, tensor: &'a SharedTensor) -> &'a Tensor {
        &tensor.0
    }

    pub fn alloc(&self, shape: &[i64], kind: Kind) -> Tensor {
        let ne = shape.iter().rev().cloned().collect::<Vec<_>>();
        let tensor = unsafe {
            bindings::ggml_new_tensor(
                self.ctx.as_ptr(),
                kind as _,
                ne.len() as _,
                ne.as_ptr() as _,
            )
        };

        Tensor {
            ctx: self.ctx,
            ptr: tensor,
        }
    }
}

pub struct SharedTensor(Tensor);

#[derive(Debug)]
pub struct Tensor {
    ctx: NonNull<bindings::ggml_context>,
    ptr: *mut bindings::ggml_tensor,
}

impl Tensor {
    #[inline]
    pub fn overhead() -> usize {
        unsafe { ggml_tensor_overhead() }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        let tensor =
            unsafe { bindings::ggml_mul_mat(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let tensor =
            unsafe { bindings::ggml_mul(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn mul_inplace(self, other: &Self) -> Self {
        let tensor = unsafe {
            bindings::ggml_mul_inplace(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr)
        };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let tensor =
            unsafe { bindings::ggml_sub(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn sum(&self) -> Self {
        let tensor = unsafe { bindings::ggml_sum(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }
    pub fn silu(&self) -> Self {
        let tensor = unsafe { bindings::ggml_silu(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn silu_inplace(self) -> Self {
        let tensor = unsafe { bindings::ggml_silu_inplace(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn transpose(&self) -> Self {
        self.transpose_view().cont()
    }

    pub fn transpose_view(&self) -> Self {
        let tensor = unsafe { bindings::ggml_transpose(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn cont(&self) -> Self {
        let tensor = unsafe { bindings::ggml_cont(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn cast(&self, kind: Kind) -> Self {
        let tensor = unsafe { bindings::ggml_cast(self.ctx.as_ptr(), &mut *self.ptr, kind as _) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn kind(&self) -> Kind {
        let inner = unsafe { &*self.ptr };
        Kind::from(inner.type_)
    }

    pub fn shape(&self) -> Vec<i64> {
        let n_dim = unsafe { ggml_n_dims(self.ptr) };
        let inner = unsafe { &*self.ptr };
        inner.ne.iter().take(n_dim as _).rev().cloned().collect()
    }
}

impl Tensor {
    fn set_data(&self, data: &[u8]) -> Result<(), Error> {
        let inner = unsafe { &mut *self.ptr };
        let numel = inner.ne.iter().product::<i64>() as usize;
        let inner_size = unsafe { ggml_type_size(inner.type_) * numel };
        if data.len() != inner_size {
            return Err(Error::DimensionMismatch(data.len(), inner_size));
        }
        unsafe {
            inner
                .data
                .copy_from_nonoverlapping(data.as_ptr() as _, data.len())
        };
        Ok(())
    }

    fn get_data(&self) -> &[u8] {
        let inner = unsafe { &*self.ptr };
        let numel = inner.ne.iter().product::<i64>() as usize;
        let inner_size = unsafe { ggml_type_size(inner.type_) * numel };
        unsafe { std::slice::from_raw_parts(inner.data as *const u8, inner_size) }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let ptr = unsafe { ggml_dup_tensor(self.ctx.as_ptr(), self.ptr) };
        Self { ctx: self.ctx, ptr }
    }
}

unsafe impl Send for Tensor {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    DimensionMismatch(usize, usize),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch(got, expected) => {
                write!(f, "Dimension mismatch: got {}, expected {}", got, expected)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn matmul<const N: usize>(a: &[f32; N], b: &[f32; N]) -> Vec<f32> {
        let mut c = Vec::with_capacity(N * N); // C^T = A * B^T
        for &j in b.iter().take(N) {
            for &i in a.iter().take(N) {
                c.push(i * j);
            }
        }
        c
    }

    fn slice_to_u8<T>(s: &[T]) -> &[u8] {
        let len = std::mem::size_of_val(s);
        let ptr = s.as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    #[test]
    fn test_tensor_mul() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = Context::new(1024 * 1024);

        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 5.0, 6.0];

        let expected_c = matmul(&a, &b);

        let mut graph = ctx.create_graph(|allocator| {
            let tensor_a = allocator.alloc(&[3, 1], Kind::F32);
            let tensor_b = allocator.alloc(&[3, 1], Kind::F32);
            let tensor_c = tensor_a.matmul(&tensor_b);
            ([tensor_a, tensor_b], tensor_c)
        });

        let inputs_shape = graph.inputs_shape();
        let output_shape = graph.output_shape();

        assert_eq!(inputs_shape, [&[3, 1], &[3, 1]]);
        assert_eq!(output_shape, &[3, 3]);

        let output = graph.compute([slice_to_u8(&a), slice_to_u8(&b)], 1)?;

        assert_eq!(output, slice_to_u8(&expected_c));

        for _ in 0..32 {
            let mut a: [f32; 3] = [0.0; 3];
            let mut b: [f32; 3] = [0.0; 3];
            for i in 0..3 {
                a[i] = rand::random();
                b[i] = rand::random();
            }

            let expected_c = matmul(&a, &b);

            let output = graph.compute([slice_to_u8(&a), slice_to_u8(&b)], 1)?;

            assert_eq!(output, slice_to_u8(&expected_c));
        }

        Ok(())
    }
}
