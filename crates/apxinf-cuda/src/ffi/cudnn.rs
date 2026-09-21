//! cuDNN convolution provider ABI. Only safe convolution plans call these APIs.
#![allow(non_snake_case)]
use std::ffi::{c_int, c_void};
pub type Descriptor = *mut c_void;
extern "C" {
    pub fn cudnnCreate(handle: *mut Descriptor) -> c_int;
    pub fn cudnnDestroy(handle: Descriptor) -> c_int;
    pub fn cudnnSetStream(handle: Descriptor, stream: *mut c_void) -> c_int;
    pub fn cudnnCreateTensorDescriptor(desc: *mut Descriptor) -> c_int;
    pub fn cudnnDestroyTensorDescriptor(desc: Descriptor) -> c_int;
    pub fn cudnnSetTensor4dDescriptor(
        desc: Descriptor,
        format: c_int,
        dtype: c_int,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> c_int;
    pub fn cudnnCreateFilterDescriptor(desc: *mut Descriptor) -> c_int;
    pub fn cudnnDestroyFilterDescriptor(desc: Descriptor) -> c_int;
    pub fn cudnnSetFilter4dDescriptor(
        desc: Descriptor,
        dtype: c_int,
        format: c_int,
        k: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> c_int;
    pub fn cudnnCreateConvolutionDescriptor(desc: *mut Descriptor) -> c_int;
    pub fn cudnnDestroyConvolutionDescriptor(desc: Descriptor) -> c_int;
    pub fn cudnnSetConvolution2dDescriptor(
        desc: Descriptor,
        ph: c_int,
        pw: c_int,
        sh: c_int,
        sw: c_int,
        dh: c_int,
        dw: c_int,
        mode: c_int,
        compute: c_int,
    ) -> c_int;
    pub fn cudnnSetConvolutionMathType(desc: Descriptor, math: c_int) -> c_int;
    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: Descriptor,
        x: Descriptor,
        w: Descriptor,
        conv: Descriptor,
        y: Descriptor,
        algo: c_int,
        bytes: *mut usize,
    ) -> c_int;
    pub fn cudnnConvolutionForward(
        handle: Descriptor,
        alpha: *const f32,
        xd: Descriptor,
        x: *const c_void,
        wd: Descriptor,
        w: *const c_void,
        conv: Descriptor,
        algo: c_int,
        workspace: *mut c_void,
        bytes: usize,
        beta: *const f32,
        yd: Descriptor,
        y: *mut c_void,
    ) -> c_int;
    pub fn cudnnTransformTensor(
        handle: Descriptor,
        alpha: *const f32,
        xd: Descriptor,
        x: *const c_void,
        beta: *const f32,
        yd: Descriptor,
        y: *mut c_void,
    ) -> c_int;
}
