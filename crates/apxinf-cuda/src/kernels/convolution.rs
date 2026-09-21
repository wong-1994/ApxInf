//! Prepared BF16 NCHW convolution with FP32 accumulation and NHWC output.
//! cuDNN remains a provider beneath this safe, model-neutral interface.
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

/// Fixed valid (zero-padding), unit-dilation 2D convolution dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Conv2dShape {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
}
impl Conv2dShape {
    fn dimensions(self) -> Result<([usize; 4], [usize; 4], [usize; 4])> {
        let x = [self.batch, self.channels, self.height, self.width];
        let w = [
            self.output_channels,
            self.channels,
            self.kernel,
            self.kernel,
        ];
        for d in x.into_iter().chain(w).chain([self.stride]) {
            if d == 0 || d > i32::MAX as usize {
                return Err(Error::Other("invalid conv2d dimension".into()));
            }
        }
        if self.kernel > self.height || self.kernel > self.width {
            return Err(Error::Other("conv2d kernel exceeds input".into()));
        }
        let y = [
            self.batch,
            (self.height - self.kernel) / self.stride + 1,
            (self.width - self.kernel) / self.stride + 1,
            self.output_channels,
        ];
        for dims in [&x, &w, &y] {
            super::contracts::checked_bytes(DType::BF16, dims, "conv2d")?;
        }
        Ok((x, w, y))
    }
}

pub struct Conv2dPlan {
    x: [usize; 4],
    w: [usize; 4],
    y: [usize; 4],
    device: usize,
    #[cfg(apxinf_cudnn)]
    native: std::sync::Mutex<Native>,
}
impl Conv2dPlan {
    /// Call before capture. Descriptor/handle creation is forbidden during capture.
    pub fn prepare(ctx: &CudaContext, shape: Conv2dShape) -> Result<Self> {
        let (x, w, y) = shape.dimensions()?;
        if !crate::workspace::may_prepare_native_resources() {
            return Err(Error::Other(
                "conv2d preparation is forbidden in capture".into(),
            ));
        }
        #[cfg(apxinf_cudnn)]
        {
            Ok(Self {
                x,
                w,
                y,
                device: ctx.device_id(),
                native: std::sync::Mutex::new(Native::new(ctx, shape, x, w, y)?),
            })
        }
        #[cfg(not(apxinf_cudnn))]
        {
            let _ = (ctx, x, w, y);
            Err(Error::Other(
                "BF16 conv2d requires building with CUDNN_LIB_DIR (cuDNN 9)".into(),
            ))
        }
    }
    pub fn workspace_bytes(&self) -> usize {
        #[cfg(apxinf_cudnn)]
        {
            self.native.lock().expect("conv2d plan poisoned").bytes
                + self.y.iter().product::<usize>() * 4
                + 512
        }
        #[cfg(not(apxinf_cudnn))]
        {
            0
        }
    }
    /// Input NCHW and weight OIHW are contiguous BF16; output is contiguous NHWC.
    /// All scratch/output allocations use the active graph workspace.
    pub fn execute(&self, ctx: &CudaContext, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        if ctx.device_id() != self.device
            || x.dtype() != DType::BF16
            || w.dtype() != DType::BF16
            || x.shape().dims() != self.x
            || w.shape().dims() != self.w
        {
            return Err(Error::Other(
                "conv2d input/weight shape, dtype or device mismatch".into(),
            ));
        }
        let xb = CudaBuffer::from_tensor(x).map_err(Error::Cuda)?;
        let wb = CudaBuffer::from_tensor(w).map_err(Error::Cuda)?;
        super::contracts::require_buffers(
            ctx,
            "conv2d",
            &[
                ("input", &xb, x.size_in_bytes()),
                ("weight", &wb, w.size_in_bytes()),
            ],
        )?;
        #[cfg(apxinf_cudnn)]
        {
            let p = self
                .native
                .lock()
                .map_err(|_| Error::Other("conv2d plan poisoned".into()))?;
            if p.stream != ctx.stream().handle() as usize {
                return Err(Error::Other("conv2d plan belongs to another stream".into()));
            }
            let size = self.y.iter().product::<usize>() * 2;
            let raw = crate::workspace::output_buffer(ctx, size)?;
            let output = crate::workspace::output_buffer(ctx, size)?;
            let scratch = crate::workspace::output_buffer(ctx, p.bytes.max(1))?;
            unsafe {
                check(crate::ffi::cudnn::cudnnConvolutionForward(
                    p.h,
                    &1.,
                    p.x,
                    xb.ptr(),
                    p.w,
                    wb.ptr(),
                    p.c,
                    0,
                    scratch.ptr(),
                    p.bytes,
                    &0.,
                    p.y,
                    raw.ptr(),
                ))?;
                check(crate::ffi::cudnn::cudnnTransformTensor(
                    p.h,
                    &1.,
                    p.y,
                    raw.ptr(),
                    &0.,
                    p.out,
                    output.ptr(),
                ))?;
            }
            Ok(output.into_tensor(Shape::new(self.y.to_vec()), DType::BF16))
        }
        #[cfg(not(apxinf_cudnn))]
        {
            Err(Error::Other("cuDNN support not compiled in".into()))
        }
    }
}
#[cfg(apxinf_cudnn)]
fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Cuda(format!("cuDNN convolution status {status}")))
    }
}
#[cfg(apxinf_cudnn)]
struct Native {
    h: crate::ffi::cudnn::Descriptor,
    x: crate::ffi::cudnn::Descriptor,
    w: crate::ffi::cudnn::Descriptor,
    c: crate::ffi::cudnn::Descriptor,
    y: crate::ffi::cudnn::Descriptor,
    out: crate::ffi::cudnn::Descriptor,
    stream: usize,
    bytes: usize,
}
// Each native handle is used only through the plan's mutex, on its original stream.
#[cfg(apxinf_cudnn)]
unsafe impl Send for Native {}
#[cfg(apxinf_cudnn)]
impl Native {
    fn new(
        ctx: &CudaContext,
        s: Conv2dShape,
        x: [usize; 4],
        w: [usize; 4],
        y: [usize; 4],
    ) -> Result<Self> {
        use crate::ffi::cudnn::*;
        let null = std::ptr::null_mut();
        let mut p = Self {
            h: null,
            x: null,
            w: null,
            c: null,
            y: null,
            out: null,
            stream: ctx.stream().handle() as usize,
            bytes: 0,
        };
        unsafe {
            check(cudnnCreate(&mut p.h))?;
            check(cudnnSetStream(p.h, ctx.stream().handle()))?;
            check(cudnnCreateTensorDescriptor(&mut p.x))?;
            check(cudnnCreateTensorDescriptor(&mut p.y))?;
            check(cudnnCreateTensorDescriptor(&mut p.out))?;
            check(cudnnCreateFilterDescriptor(&mut p.w))?;
            check(cudnnCreateConvolutionDescriptor(&mut p.c))?;
            check(cudnnSetTensor4dDescriptor(
                p.x, 0, 9, x[0] as _, x[1] as _, x[2] as _, x[3] as _,
            ))?;
            check(cudnnSetTensor4dDescriptor(
                p.y, 0, 9, y[0] as _, y[3] as _, y[1] as _, y[2] as _,
            ))?;
            check(cudnnSetTensor4dDescriptor(
                p.out, 1, 9, y[0] as _, y[3] as _, y[1] as _, y[2] as _,
            ))?;
            check(cudnnSetFilter4dDescriptor(
                p.w, 9, 0, w[0] as _, w[1] as _, w[2] as _, w[3] as _,
            ))?;
            check(cudnnSetConvolution2dDescriptor(
                p.c,
                0,
                0,
                s.stride as _,
                s.stride as _,
                1,
                1,
                1,
                0,
            ))?;
            check(cudnnSetConvolutionMathType(p.c, 0))?;
            check(cudnnGetConvolutionForwardWorkspaceSize(
                p.h,
                p.x,
                p.w,
                p.c,
                p.y,
                0,
                &mut p.bytes,
            ))?;
        }
        Ok(p)
    }
}
#[cfg(apxinf_cudnn)]
impl Drop for Native {
    fn drop(&mut self) {
        unsafe {
            use crate::ffi::cudnn::*;
            if !self.x.is_null() {
                cudnnDestroyTensorDescriptor(self.x);
            }
            if !self.y.is_null() {
                cudnnDestroyTensorDescriptor(self.y);
            }
            if !self.out.is_null() {
                cudnnDestroyTensorDescriptor(self.out);
            }
            if !self.w.is_null() {
                cudnnDestroyFilterDescriptor(self.w);
            }
            if !self.c.is_null() {
                cudnnDestroyConvolutionDescriptor(self.c);
            }
            if !self.h.is_null() {
                cudnnDestroy(self.h);
            }
        }
    }
}
