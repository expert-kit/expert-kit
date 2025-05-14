use ek_base::config::OpenDALStorage;
use opendal::{
    Operator,
    services::{Fs, S3 as opendal_s3},
};

pub fn op_from_settings(config: &OpenDALStorage) -> opendal::Operator {
    match config {
        OpenDALStorage::S3(s3_cfg) => {
            let builder = opendal_s3::default()
                .access_key_id(s3_cfg.access_key_id.as_str())
                .secret_access_key(s3_cfg.access_key_secret.as_str())
                .endpoint(s3_cfg.endpoint.as_str())
                .region(s3_cfg.region.as_str());
            Operator::new(builder).unwrap().finish()
        }
        OpenDALStorage::Fs(fs_cfg) => {
            let builder = Fs::default().root(&fs_cfg.path);
            Operator::new(builder).unwrap().finish()
        }
    }
}
