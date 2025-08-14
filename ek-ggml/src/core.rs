use std::{cell::OnceCell, ptr::NonNull};

use crate::bindings::{ggml_dup_tensor, ggml_n_dims, ggml_type_size};

#[allow(warnings)]
pub(crate) mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug)]
pub struct Context {
    ptr: NonNull<bindings::ggml_context>,
}

thread_local! {
    static CONTEXT: OnceCell<Context> = OnceCell::new();
}

impl Context {
    pub fn init(size: usize) {
        CONTEXT.with(|ctx| {
            ctx.get_or_init(|| Context::new(size));
        });
    }

    fn new(size: usize) -> Self {
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
}

unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { bindings::ggml_free(self.ptr.as_ptr()) };
    }
}

pub struct Graph {
    ctx: NonNull<bindings::ggml_context>,
    ptr: NonNull<bindings::ggml_cgraph>,
}

impl Graph {
    pub fn new() -> Self {
        CONTEXT.with(|ctx| {
            let ctx = ctx.get().expect("Context not initialized");
            let graph = unsafe { bindings::ggml_new_graph(ctx.ptr.as_ptr()) };

            Self {
                ctx: NonNull::new(ctx.ptr.as_ptr()).unwrap(),
                ptr: NonNull::new(graph).unwrap(),
            }
        })
    }

    pub fn build_forward(&mut self, tensor: &Tensor) {
        unsafe { bindings::ggml_build_forward_expand(self.ptr.as_mut(), tensor.ptr) };
    }

    pub fn compute(&mut self, n_threads: usize) {
        unsafe {
            bindings::ggml_graph_compute_with_ctx(
                self.ctx.as_ptr(),
                self.ptr.as_mut(),
                n_threads as _,
            )
        };
    }
}

unsafe impl Send for Graph {}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    F32 = bindings::ggml_type_GGML_TYPE_F32,
}

impl Kind {
    pub fn size(&self) -> usize {
        let ggml_type: bindings::ggml_type = *self as _;
        unsafe { bindings::ggml_type_size(ggml_type) }
    }
}

impl From<bindings::ggml_type> for Kind {
    fn from(ggml_type: bindings::ggml_type) -> Self {
        match ggml_type {
            bindings::ggml_type_GGML_TYPE_F32 => Kind::F32,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    ctx: NonNull<bindings::ggml_context>,
    ptr: *mut bindings::ggml_tensor,
}

impl Tensor {
    pub fn empty(shape: &[i64], kind: Kind) -> Option<Self> {
        CONTEXT.with(|ctx| {
            let ctx = ctx.get().expect("Context not initialized");
            let ne: Vec<i64> = shape.iter().rev().cloned().collect();
            let tensor = unsafe {
                bindings::ggml_new_tensor(
                    ctx.ptr.as_ptr(),
                    kind as _,
                    ne.len() as _,
                    ne.as_ptr() as _,
                )
            };
            Some(Self {
                ctx: NonNull::new(ctx.ptr.as_ptr())?,
                ptr: tensor,
            })
        })
    }

    pub unsafe fn from_raw(data: &[u8], shape: &[i64], kind: Kind) -> Option<Self> {
        CONTEXT.with(|ctx| {
            let ctx = ctx.get().expect("Context not initialized");
            let ne: Vec<i64> = shape.iter().rev().cloned().collect();
            let tensor = unsafe {
                &mut *bindings::ggml_new_tensor(
                    ctx.ptr.as_ptr(),
                    kind as _,
                    ne.len() as _,
                    ne.as_ptr() as *const _,
                )
            };

            unsafe {
                tensor
                    .data
                    .copy_from_nonoverlapping(data.as_ptr() as _, data.len())
            };

            Some(Self {
                ctx: NonNull::new(ctx.ptr.as_ptr()).unwrap(),
                ptr: tensor,
            })
        })
    }

    pub fn set_data(&mut self, data: &[u8]) -> Result<(), (usize, usize)> {
        let inner = unsafe { &mut *self.ptr };
        let inner_size =
            unsafe { ggml_type_size(inner.type_) * inner.ne.iter().product::<i64>() as usize };
        if data.len() != inner_size {
            return Err((data.len(), inner_size));
        }
        unsafe {
            inner
                .data
                .copy_from_nonoverlapping(data.as_ptr() as _, data.len())
        };
        Ok(())
    }

    pub fn shape(&self) -> Vec<i64> {
        let n_dim = unsafe { ggml_n_dims(self.ptr) };
        let inner = unsafe { &*self.ptr };
        inner.ne.iter().take(n_dim as _).rev().cloned().collect()
    }

    pub fn raw_data(&self) -> &[u8] {
        let inner = unsafe { &*self.ptr };
        let numel = inner.ne.iter().product::<i64>() as usize;
        unsafe {
            std::slice::from_raw_parts(inner.data as *const u8, ggml_type_size(inner.type_) * numel)
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let tensor =
            unsafe { bindings::ggml_mul_mat(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let tensor =
            unsafe { bindings::ggml_mul(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let tensor =
            unsafe { bindings::ggml_sub(self.ctx.as_ptr(), &mut *self.ptr, &mut *other.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn sum(&self) -> Tensor {
        let tensor = unsafe { bindings::ggml_sum(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }
    pub fn silu(&self) -> Tensor {
        let tensor = unsafe { bindings::ggml_silu(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn transpose(&self) -> Tensor {
        self.transpose_view().cont()
    }

    pub fn transpose_view(&self) -> Tensor {
        let tensor = unsafe { bindings::ggml_transpose(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn cont(&self) -> Tensor {
        let tensor = unsafe { bindings::ggml_cont(self.ctx.as_ptr(), &mut *self.ptr) };
        Self {
            ctx: self.ctx,
            ptr: tensor,
        }
    }

    pub fn cast(&self, kind: Kind) -> Tensor {
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
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let ptr = unsafe { ggml_dup_tensor(self.ctx.as_ptr(), &mut *self.ptr) };
        Self { ctx: self.ctx, ptr }
    }
}

unsafe impl Send for Tensor {}

#[cfg(test)]
mod test {
    use super::*;

    fn matmul<const N: usize>(a: &[f32; N], b: &[f32; N]) -> Vec<f32> {
        let mut c = Vec::with_capacity(N * N); // C^T = A * B^T
        for j in 0..N {
            for i in 0..N {
                c.push(a[i] * b[j]);
            }
        }
        c
    }

    fn tensor(a: &[f32]) -> Tensor {
        let len = a.len();
        let a = unsafe {
            std::slice::from_raw_parts_mut(
                a.as_ptr() as *mut _,
                a.len() * std::mem::size_of::<f32>(),
            )
        };
        unsafe { Tensor::from_raw(a, &[len as _, 1], Kind::F32).unwrap() }
    }

    #[test]
    fn test_tensor_mul() -> Result<(), Box<dyn std::error::Error>> {
        Context::init(1024 * 1024);

        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 5.0, 6.0];

        let expected_c = matmul(&a, &b);

        let mut tensor_a = tensor(&a);
        let mut tensor_b = tensor(&b);

        assert_eq!(tensor_a.shape(), &[3, 1]);
        assert_eq!(tensor_b.shape(), &[3, 1]);
        let mut tensor_c = tensor_a.matmul(&mut tensor_b);

        let mut graph = Graph::new();
        graph.build_forward(&mut tensor_c);
        graph.compute(1);
        let c_data = tensor_c.raw_data();
        let result = unsafe {
            std::slice::from_raw_parts_mut(
                c_data.as_ptr() as *mut f32,
                c_data.len() / std::mem::size_of::<f32>(),
            )
        };

        assert_eq!(result, &expected_c);

        for _ in 0..32 {
            let mut a: [f32; 3] = [0.0; 3];
            let mut b: [f32; 3] = [0.0; 3];
            for i in 0..3 {
                a[i] = rand::random();
                b[i] = rand::random();
            }

            let expected_c = matmul(&a, &b);

            let a = unsafe {
                std::slice::from_raw_parts_mut(
                    a.as_ptr() as _,
                    a.len() * std::mem::size_of::<f32>(),
                )
            };
            let b = unsafe {
                std::slice::from_raw_parts_mut(
                    b.as_ptr() as _,
                    b.len() * std::mem::size_of::<f32>(),
                )
            };

            tensor_a.set_data(a).unwrap();
            tensor_b.set_data(b).unwrap();

            graph.compute(1);
            let c_data = tensor_c.raw_data();
            let result = unsafe {
                std::slice::from_raw_parts_mut(
                    c_data.as_ptr() as *mut f32,
                    c_data.len() / std::mem::size_of::<f32>(),
                )
            };

            assert_eq!(result, &expected_c);
        }

        Ok(())
    }
}
