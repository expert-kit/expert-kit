use opendal;
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

    #[error("opendal error")]
    OpenDALError(opendal::Error),
}

pub type EKResult<T> = std::result::Result<T, EKError>;

impl From<opendal::Error> for EKError {
    fn from(value: opendal::Error) -> Self {
        return Self::OpenDALError(value);
    }
}
