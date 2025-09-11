use std::{pin::Pin, ptr::NonNull, rc::Rc};

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
            no_alloc: true,
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
        data: Box<[u8]>,
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

        let tensor = TensorNode::new_with_data_set(self.ptr, tensor, &[], Some(Pin::new(data)));

        Ok(SharedTensor(tensor))
    }

    pub fn create_graph<const N: usize>(
        &self,
        compute: impl FnOnce(&TensorAllocator) -> Result<([TensorNode; N], TensorNode), Error>,
    ) -> Result<Graph<N>, Error> {
        let graph = unsafe { bindings::ggml_new_graph(self.ptr.as_ptr()) };

        let allocator = TensorAllocator { ctx: self.ptr };
        let (inputs, output) = compute(&allocator)?;

        unsafe { bindings::ggml_build_forward_expand(graph, output.ptr) };

        Ok(Graph {
            ctx: NonNull::new(self.ptr.as_ptr()).unwrap(),
            ptr: NonNull::new(graph).unwrap(),
            inputs,
            output,
        })
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
    inputs: [TensorNode; N],
    output: TensorNode,
}

impl<const N: usize> Graph<N> {
    pub fn overhead() -> usize {
        unsafe { bindings::ggml_graph_overhead() }
    }

    pub fn default_size() -> usize {
        bindings::GGML_DEFAULT_GRAPH_SIZE as _
    }

    pub fn inputs_kind(&self) -> [Kind; N] {
        self.inputs.each_ref().map(|tensor| tensor.kind())
    }

    pub fn output_kind(&self) -> Kind {
        self.output.kind()
    }

    pub fn inputs_shape(&self) -> [Vec<i64>; N] {
        self.inputs.each_ref().map(|tensor| tensor.shape())
    }

    pub fn output_shape(&self) -> Vec<i64> {
        self.output.shape()
    }

    pub fn compute(&mut self, inputs: [&[u8]; N], n_threads: usize) -> Result<Vec<u8>, Error> {
        for (tensor, &data) in self.inputs.iter_mut().zip(inputs.iter()) {
            if tensor.data.is_some() {
                return Err(Error::InputTensorDataFound);
            }
            unsafe { (*tensor.ptr).data = data.as_ptr() as _ };
        }

        let mut result = vec![0u8; self.output.size()];

        if let Some(place) = &self.output.data {
            unsafe {
                bindings::ggml_graph_compute_with_ctx(
                    self.ctx.as_ptr(),
                    self.ptr.as_mut(),
                    n_threads as _,
                )
            };

            result.copy_from_slice(place);
        } else {
            return Err(Error::OutputTensorDataNotFound);
        }

        Ok(result)
    }

    pub fn print(&self) {
        unsafe { bindings::ggml_graph_print(self.ptr.as_ptr()) };
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
    pub fn borrow(&self, tensor: &SharedTensor) -> TensorNode {
        tensor.0.clone()
    }

    pub fn alloc(&self, shape: &[i64], kind: Kind) -> TensorNode {
        let ne = shape.iter().rev().cloned().collect::<Vec<_>>();
        let tensor = unsafe {
            bindings::ggml_new_tensor(
                self.ctx.as_ptr(),
                kind as _,
                ne.len() as _,
                ne.as_ptr() as _,
            )
        };

        TensorNode {
            inner: Rc::new(Tensor {
                ctx: self.ctx,
                ptr: tensor,
                data: None,
            }),
            _deps: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    ctx: NonNull<bindings::ggml_context>,
    ptr: *mut bindings::ggml_tensor,
    data: Option<Pin<Box<[u8]>>>,
}

impl Tensor {
    #[inline]
    pub fn overhead() -> usize {
        unsafe { bindings::ggml_tensor_overhead() }
    }

    pub fn kind(&self) -> Kind {
        let inner = unsafe { &*self.ptr };
        Kind::from(inner.type_)
    }

    pub fn shape(&self) -> Vec<i64> {
        let n_dim = unsafe { bindings::ggml_n_dims(self.ptr) };
        let inner = unsafe { &*self.ptr };
        inner.ne.iter().take(n_dim as _).rev().cloned().collect()
    }

    pub fn size(&self) -> usize {
        self.shape().iter().product::<i64>() as usize * self.kind().size()
    }
}

pub struct SharedTensor(TensorNode);

#[derive(Debug, Clone)]
pub struct TensorNode {
    inner: Rc<Tensor>,
    _deps: Vec<TensorNode>,
}

impl TensorNode {
    pub fn name(&mut self, name: &str) {
        unsafe { bindings::ggml_set_name(self.inner.ptr, name.as_bytes().as_ptr() as _) };
    }

    pub fn matmul(&self, other: &Self) -> Self {
        let tensor = unsafe {
            bindings::ggml_mul_mat(self.inner.ctx.as_ptr(), self.inner.ptr, other.inner.ptr)
        };
        Self::new_with_data_alloc(self.inner.ctx, tensor, &[self.clone(), other.clone()])
    }

    pub fn mul(&self, other: &Self) -> Self {
        let tensor =
            unsafe { bindings::ggml_mul(self.inner.ctx.as_ptr(), self.inner.ptr, other.inner.ptr) };
        Self::new_with_data_alloc(self.inner.ctx, tensor, &[self.clone(), other.clone()])
    }

    pub fn mul_inplace(self, other: &Self) -> Result<Self, Error> {
        if let Ok(inner) = Rc::try_unwrap(self.inner) {
            let tensor = unsafe {
                bindings::ggml_mul_inplace(inner.ctx.as_ptr(), inner.ptr, other.inner.ptr)
            };
            Ok(Self::new_with_data_set(
                inner.ctx,
                tensor,
                std::slice::from_ref(&other.clone()),
                inner.data,
            ))
        } else {
            Err(Error::TensorNotOwned)
        }
    }

    pub fn silu(&self) -> Self {
        let tensor = unsafe { bindings::ggml_silu(self.inner.ctx.as_ptr(), self.inner.ptr) };
        Self::new_with_data_alloc(self.inner.ctx, tensor, std::slice::from_ref(self))
    }

    pub fn silu_inplace(self) -> Result<Self, Error> {
        if let Ok(inner) = Rc::try_unwrap(self.inner) {
            let tensor = unsafe { bindings::ggml_silu_inplace(inner.ctx.as_ptr(), inner.ptr) };
            Ok(Self::new_with_data_set(inner.ctx, tensor, &[], inner.data))
        } else {
            Err(Error::TensorNotOwned)
        }
    }

    pub fn transpose(&self) -> Self {
        let tensor = unsafe { bindings::ggml_transpose(self.inner.ctx.as_ptr(), self.inner.ptr) };
        let tensor = unsafe { bindings::ggml_cont(self.inner.ctx.as_ptr(), tensor) };
        Self::new_with_data_alloc(self.inner.ctx, tensor, std::slice::from_ref(self))
    }

    pub fn cast(&self, kind: Kind) -> Self {
        let tensor =
            unsafe { bindings::ggml_cast(self.inner.ctx.as_ptr(), self.inner.ptr, kind as _) };
        Self::new_with_data_alloc(self.inner.ctx, tensor, std::slice::from_ref(self))
    }
}

impl TensorNode {
    fn new_with_data_set(
        ctx: NonNull<bindings::ggml_context>,
        ptr: *mut bindings::ggml_tensor,
        deps: &[TensorNode],
        data: Option<Pin<Box<[u8]>>>,
    ) -> Self {
        let mut tensor = Tensor {
            ctx,
            ptr,
            data: None,
        };

        if let Some(data) = &data {
            unsafe { (*ptr).data = data.as_ptr() as _ };
        }

        tensor.data = data;

        Self {
            inner: Rc::new(tensor),
            _deps: deps.to_vec(),
        }
    }

    fn new_with_data_alloc(
        ctx: NonNull<bindings::ggml_context>,
        ptr: *mut bindings::ggml_tensor,
        deps: &[TensorNode],
    ) -> Self {
        let mut tensor = Tensor {
            ctx,
            ptr,
            data: None,
        };

        let size = tensor.size();
        let data = Pin::new(vec![0u8; size].into_boxed_slice());

        unsafe { (*ptr).data = data.as_ptr() as _ };

        tensor.data = Some(data);

        Self {
            inner: Rc::new(tensor),
            _deps: deps.to_vec(),
        }
    }
}

impl std::ops::Deref for TensorNode {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    DimensionMismatch(usize, usize),
    InputTensorDataFound,
    OutputTensorDataNotFound,
    TensorNotOwned,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch(got, expected) => {
                write!(f, "Dimension mismatch: got {}, expected {}", got, expected)
            }
            Error::InputTensorDataFound => {
                write!(f, "Input tensor data found")
            }
            Error::OutputTensorDataNotFound => {
                write!(f, "Output tensor data not found")
            }
            Error::TensorNotOwned => {
                write!(f, "Tensor is not owned")
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

        let mut graph = ctx
            .create_graph(|allocator| {
                let tensor_a = allocator.alloc(&[3, 1], Kind::F32);
                let tensor_b = allocator.alloc(&[3, 1], Kind::F32);
                let tmp = tensor_a.matmul(&tensor_b);
                let tmp = tmp.transpose();
                let tmp = tmp.transpose();
                Ok(([tensor_a, tensor_b], tmp))
            })
            .unwrap();

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
