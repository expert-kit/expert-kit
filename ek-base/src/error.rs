use thiserror::Error;

#[derive(Error, Debug)]
pub enum EKError {
    #[error("error related to safe tensor conversion")]
    SafeTensorError,
}
