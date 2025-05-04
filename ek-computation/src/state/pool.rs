use ek_base::config::get_config_key;
use once_cell::sync::Lazy;

use diesel_async::AsyncPgConnection;
use diesel_async::pooled_connection::AsyncDieselConnectionManager;
use diesel_async::pooled_connection::deadpool::Pool;

pub static POOL: Lazy<Pool<AsyncPgConnection>> = Lazy::new(|| {
    let url = get_config_key("db_dsn");
    log::debug!("connect to database {}", url);
    let config = AsyncDieselConnectionManager::<diesel_async::AsyncPgConnection>::new(url);
    Pool::builder(config).build().unwrap()
});
