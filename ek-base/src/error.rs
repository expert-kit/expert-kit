use opendal;
use std::string;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EKError {
    #[error("error related to safe tensor conversion")]
    SafeTensorError(#[from] safetensors::SafeTensorError),

    #[error("error related tch-rs")]
    TchError(#[from] tch::TchError),

    #[error("tensor name not found")]
    SafeTensorNotFound,

    #[error("expert not found in the computation node")]
    ExpertNotFound(string::String),

    #[error("opendal error")]
    OpenDALError(#[from] opendal::Error),
}

pub type EKResult<T> = std::result::Result<T, EKError>;
