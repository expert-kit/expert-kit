use actix_web::{ResponseError, http::header::HeaderValue};
use deadpool::PoolError;
use diesel;
use diesel_async::pooled_connection::deadpool;
use opendal;
use std::{fmt::Write, string};
use tokio::task::JoinError;
use tonic::Status;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EKError {
    #[error("error related to safe tensor conversion `{0}`")]
    SafeTensorError(#[from] safetensors::SafeTensorError),

    #[error("error related to tch-rs")]
    TchError(#[from] Box<tch::TchError>),

    #[error("tonic errors")]
    TonicError(Box<tonic::Status>),

    #[error("tensor name not found")]
    SafeTensorNotFound,

    #[error("expert not found in the computation node")]
    ExpertNotFound(string::String),

    #[error("expert weight not found in tensor bundle")]
    ExpertWeightNotFound(string::String),

    #[error("NotFound `{0}`")]
    NotFound(string::String),

    #[error("opendal error {0}")]
    OpenDALError(#[from] Box<opendal::Error>),

    #[error("diesel error: {0}")]
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
    TonicTransportError(#[from] Box<tonic::transport::Error>),

    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("json error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("parse int error")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("reqwest error {0}")]
    ReqwestError(#[from] Box<reqwest::Error>),

    #[error("onnx error {0}")]
    OnnxError(#[from] Box<ort::Error>),

    #[error("OpenDAL error {0}")]
    RuntimeError(string::String),
}

pub type EKResult<T> = std::result::Result<T, EKError>;

impl From<tch::TchError> for EKError {
    fn from(err: tch::TchError) -> Self {
        EKError::TchError(Box::new(err))
    }
}

impl From<tonic::Status> for EKError {
    fn from(err: tonic::Status) -> Self {
        EKError::TonicError(Box::new(err))
    }
}

impl From<opendal::Error> for EKError {
    fn from(err: opendal::Error) -> Self {
        EKError::OpenDALError(Box::new(err))
    }
}

impl From<tonic::transport::Error> for EKError {
    fn from(err: tonic::transport::Error) -> Self {
        EKError::TonicTransportError(Box::new(err))
    }
}

impl From<reqwest::Error> for EKError {
    fn from(err: reqwest::Error) -> Self {
        EKError::ReqwestError(Box::new(err))
    }
}

impl From<ort::Error> for EKError {
    fn from(err: ort::Error) -> Self {
        EKError::OnnxError(Box::new(err))
    }
}

impl From<EKError> for Status {
    fn from(value: EKError) -> Self {
        Status::internal(value.to_string())
    }
}

impl ResponseError for EKError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        actix_web::http::StatusCode::INTERNAL_SERVER_ERROR
    }

    fn error_response(&self) -> actix_web::HttpResponse<actix_web::body::BoxBody> {
        let mut res = actix_web::HttpResponse::new(self.status_code());
        let mut buf = actix_web::web::BytesMut::new();
        let _ = buf.write_str(self.to_string().as_str());
        let mime = HeaderValue::from_static("text/plain");
        res.headers_mut()
            .insert(actix_web::http::header::CONTENT_TYPE, mime);

        res.set_body(actix_web::body::BoxBody::new(buf))
    }
}
