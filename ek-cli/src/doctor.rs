use diesel::sql_query;
use diesel_async::RunQueryDsl;
use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use ek_computation::state::pool::POOL;
use tokio::task::JoinHandle;

struct Checker {
    success: &'static str,
    failed: &'static str,
    handle: JoinHandle<Result<Option<String>, EKError>>,
}

pub struct Doctor {
    queue: Option<Vec<Checker>>,
}

impl Doctor {
    pub fn new() -> Self {
        let checkers = vec![
            Checker {
                success: "settings can be loaded",
                failed: "please check the config file and env variables",
                handle: tokio::spawn(async move {
                    let _settings = get_ek_settings();
                    Ok(None)
                }),
            },
            Checker {
                success: "database connection is ok",
                failed: "please check the config file and env variables",
                handle: tokio::spawn(async move {
                    let mut conn = POOL.get().await?;
                    sql_query("SELECT 1").execute(&mut conn).await?;
                    Ok(None)
                }),
            },
        ];
        Self {
            queue: Some(checkers),
        }
    }

    pub async fn run(&mut self) -> EKResult<()> {
        let queue = self.queue.take().unwrap();
        for checker in queue {
            let res = checker.handle.await;
            match res {
                Ok(Ok(None)) => {
                    log::info!("✅ Success: {}", checker.success);
                }

                Ok(Ok(Some(w))) => {
                    log::error!("⚠️ Warn: {}\n\thint:{}", checker.success, w,);
                }
                Ok(Err(e)) => {
                    log::error!(
                        "❌ Failed: {}\n\tpossible reason:{}\n\tsuggestion:{}",
                        checker.success,
                        e,
                        checker.failed,
                    );
                }
                Err(e) => {
                    log::error!(
                        "❌ Panic: {}\n\tpossible reason:{}\n\tsuggestion:{}",
                        checker.success,
                        e,
                        checker.failed,
                    );
                }
            };
        }
        Ok(())
    }
}

pub async fn doctor_main() -> EKResult<()> {
    let mut doc = Doctor::new();
    doc.run().await?;
    Ok(())
}
