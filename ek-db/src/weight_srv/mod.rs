mod manager;
use std::{net::ToSocketAddrs, path::PathBuf, sync::Arc, vec};

use actix_web::{App, HttpRequest, HttpServer, get, web};
use ek_base::error::EKResult;
use manager::WeightManager;
use tokio::sync::OnceCell;

#[get("/weight/{model}/{key}")]
async fn weight_load(
    req: HttpRequest,
    wm: web::Data<Arc<WeightManager<'static>>>,
) -> EKResult<Vec<u8>> {
    let model = req.match_info().get("model").unwrap();
    let key = req.match_info().get("key").unwrap();
    let tv = wm.load(model.to_owned(), key.to_owned()).await?;
    Ok(tv)
}

async fn load_manager(roots: &[PathBuf]) -> Arc<WeightManager<'static>> {
    static WM_CELL: OnceCell<Arc<WeightManager<'static>>> = OnceCell::const_new();
    let wm = WM_CELL
        .get_or_init(|| async {
            let mut valid = vec![];
            for root in roots.iter() {
                if !root.exists() {
                    log::warn!("model path not found {:?} ", root.to_str());
                    continue;
                }
                valid.push(root.clone());
            }
            for a in &valid {
                log::info!("model path found {:?}", a.to_str());
            }
            let res = WeightManager::new(&valid).await.unwrap();
            Arc::new(res)
        })
        .await;
    wm.clone()
}

pub async fn listen<A: ToSocketAddrs>(roots: &[PathBuf], addr: A) -> std::io::Result<()> {
    let wm = load_manager(roots).await;
    let addr = addr.to_socket_addrs().unwrap().collect::<Vec<_>>();
    log::info!("starting weight server.");
    for a in addr.iter() {
        log::info!("listening on {}", a);
    }
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(wm.clone()))
            .service(weight_load)
    })
    .bind(addr.as_slice())?
    .run()
    .await
}

#[cfg(test)]
mod test {
    use super::*;

    use actix_web::{App, body::to_bytes, http::header::ContentType, test};
    use ek_base::utils::workspace_root;

    #[actix_web::test]
    async fn test_index_get() {
        let root = workspace_root();
        let test_model = root.join("ek-db").join("resources").join("ds-tiny");
        let wm = load_manager(&[test_model]).await;
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(wm.clone()))
                .service(weight_load),
        )
        .await;
        let req = test::TestRequest::default()
            .uri("/weight/ds-tiny/model.layers.21.mlp.experts.94.down_proj.weight")
            .insert_header(ContentType::plaintext())
            .to_request();
        let resp = test::call_service(&app, req).await;
        let success = resp.status().is_success();
        assert!(success);
        let body = resp.into_body();
        let res = to_bytes(body).await.unwrap();
        let st = safetensors::SafeTensors::deserialize(&res).unwrap();
        let tv = st.tensor("data").unwrap();

        assert_eq!(tv.shape(), &[16, 8]);
    }
}
