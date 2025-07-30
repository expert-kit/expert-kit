use std::sync::LazyLock;

use ek_base::config::get_ek_settings;

use diesel_async::AsyncPgConnection;
use diesel_async::pooled_connection::AsyncDieselConnectionManager;
use diesel_async::pooled_connection::deadpool::Pool;

pub static POOL: LazyLock<Pool<AsyncPgConnection>> = LazyLock::new(|| {
    let settings = get_ek_settings();
    log::debug!("connect to database {}", settings.db.db_dsn.clone());
    let config = AsyncDieselConnectionManager::<diesel_async::AsyncPgConnection>::new(
        settings.db.db_dsn.clone(),
    );
    Pool::builder(config)
        .max_size(settings.db.max_conn_size)
        .build()
        .unwrap()
});
