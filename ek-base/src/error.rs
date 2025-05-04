use deadpool::PoolError;
use diesel;
use diesel_async::pooled_connection::deadpool;
use opendal;
use std::string;
use tokio::task::JoinError;
use tonic::Status;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EKError {
    #[error("error related to safe tensor conversion `{0}`")]
    SafeTensorError(#[from] safetensors::SafeTensorError),

    #[error("error related to tch-rs")]
    TchError(#[from] tch::TchError),

    #[error("tonic errors")]
    TonicError(#[from] tonic::Status),

    #[error("tensor name not found")]
    SafeTensorNotFound,

    #[error("expert not found in the computation node")]
    ExpertNotFound(string::String),

    #[error("expert weight not found in tensor bundle")]
    ExpertWeightNotFound(string::String),

    #[error("NotFound `{0}`")]
    NotFound(string::String),

    #[error("opendal error")]
    OpenDALError(#[from] opendal::Error),

    #[error("diesel error")]
    DieselError(#[from] diesel::result::Error),

    #[error("deadpool error")]
    DeadPoolError(#[from] PoolError),

    #[error("db error")]
    DBError(),

    #[error("join error")]
    TokioJoinError(#[from] JoinError),

    #[error("invalid input")]
    InvalidInput(string::String),

    #[error("tonic transport error")]
    TonicTransportError(#[from] tonic::transport::Error),
}

pub type EKResult<T> = std::result::Result<T, EKError>;

impl From<EKError> for Status {
    fn from(value: EKError) -> Self {
        Status::internal(value.to_string())
    }
}
