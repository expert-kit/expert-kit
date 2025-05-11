use diesel::sql_query;
use diesel_async::RunQueryDsl;
use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use ek_computation::state::{io::StateReaderImpl, pool::POOL};
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
        let checkers = [
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
            Checker {
                success: "instance and model exist",
                failed: "please create instance before running model.",
                handle: tokio::spawn(async move {
                    let settings = get_ek_settings();
                    let rcli = StateReaderImpl::default();
                    let res = rcli.instance_by_name(&settings.instance_name).await?;
                    if res.is_none() {
                        return Err(EKError::NotFound(format!(
                            "instance {} not found",
                            settings.instance_name
                        )));
                    }
                    Ok(None)
                }),
            },
            Checker {
                success: "Worker nodes registered to meta-db.",
                failed: "please run at least one worker node",
                handle: tokio::spawn(async move {
                    let rcli = StateReaderImpl::default();
                    let res = rcli.active_nodes().await?;
                    if res.is_empty() {
                        return Err(EKError::NotFound("no active worker.".to_string()));
                    }
                    Ok(None)
                }),
            },
        ]
        .into_iter()
        .collect::<Vec<_>>();
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
                    log::info!("✅ \tSuccess: {}", checker.success);
                }
                Ok(Ok(Some(w))) => {
                    log::error!("⚠️ \t   Warn: {}\n\thint:{}", checker.success, w,);
                }
                Ok(Err(e)) => {
                    log::error!(
                        "❌ \t Failed: {}\n\terror: {}\n\tsuggestion: {}",
                        checker.success,
                        e,
                        checker.failed,
                    );
                }
                Err(e) => {
                    log::error!(
                        "❌ \t Panic: {}\n\terror: {}\n\tsuggestion: {}",
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
