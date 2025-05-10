mod manager;
use std::{net::ToSocketAddrs, path::PathBuf};

use actix_web::{App, HttpRequest, HttpServer, get, web};
use ek_base::error::EKResult;
use manager::WeightManager;
use tokio::sync::OnceCell;

#[get("/weight/{model}/{key}")]
async fn weight_load(
    req: HttpRequest,
    wm: web::Data<&'static WeightManager<'static>>,
) -> EKResult<Vec<u8>> {
    let model = req.match_info().get("model").unwrap();
    let key = req.match_info().get("key").unwrap();
    let pretrained = wm.load_pretrained(model.to_owned()).await?;
    let tv = pretrained.read().await.get_layer(key.to_owned()).await?;
    Ok(tv)
}

async fn load_manager(roots: &'static [PathBuf]) -> &'static WeightManager<'static> {
    static WM_CELL: OnceCell<WeightManager<'static>> = OnceCell::const_new();

    (WM_CELL
        .get_or_init(|| async { WeightManager::new(roots).await.unwrap() })
        .await) as _
}

pub async fn listen<A: ToSocketAddrs>(roots: &'static [PathBuf], addr: A) -> std::io::Result<()> {
    let wm = load_manager(roots).await;
    let addr = addr.to_socket_addrs().unwrap().collect::<Vec<_>>();
    log::info!("starting weight server.");
    for a in addr.iter() {
        log::info!("listening on {}", a);
    }
    HttpServer::new(move || App::new().app_data(web::Data::new(wm)).service(weight_load))
        .bind(addr.as_slice())?
        .run()
        .await
}

#[cfg(test)]
mod test {
    use std::mem::transmute;

    use super::*;

    use actix_web::{App, body::to_bytes, http::header::ContentType, test};
    use ek_base::utils::workspace_root;

    #[actix_web::test]
    async fn test_index_get() {
        let root = workspace_root();
        let test_model: PathBuf = root.join("ek-db").join("resources").join("ds-tiny");
        let tm = vec![test_model.clone()];
        let tm: &'static [PathBuf] = unsafe { transmute(tm.as_slice()) };
        let wm = load_manager(tm).await;
        let app =
            test::init_service(App::new().app_data(web::Data::new(wm)).service(weight_load)).await;
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
