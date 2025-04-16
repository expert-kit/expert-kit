
use once_cell::sync::Lazy;

use diesel_async::pooled_connection::deadpool::Pool;
use diesel_async::pooled_connection::AsyncDieselConnectionManager;
use diesel_async::AsyncPgConnection;

pub static POOL: Lazy<Pool<AsyncPgConnection>> = Lazy::new(|| {
    let config = AsyncDieselConnectionManager::<diesel_async::AsyncPgConnection>::new(
        std::env::var("DATABASE_URL").unwrap(),
    );
    
    Pool::builder(config).build().unwrap()
});
