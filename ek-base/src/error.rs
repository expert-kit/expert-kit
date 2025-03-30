use std::string;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EKError {
    #[error("error related to safe tensor conversion")]
    SafeTensorError,

    #[error("tensor name not found")]
    SafeTensorNotFound,

    #[error("expert not found in the computation node")]
    ExpertNotFound(string::String),
}

pub type EKResult<T> = std::result::Result<T, EKError>;
