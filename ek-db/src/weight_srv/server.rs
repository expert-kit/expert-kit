use std::{net::ToSocketAddrs, path::PathBuf};

use super::manager::WeightManager;
use actix_web::{App, HttpRequest, HttpServer, Responder, get, web};
use ek_base::error::{EKError, EKResult};
use tokio::sync::OnceCell;

#[get("/meta/vital/{model}")]
async fn load_meta_vital(
    req: HttpRequest,
    wm: web::Data<&'static WeightManager<'static>>,
) -> EKResult<impl Responder> {
    let model = req.match_info().get("model").unwrap();
    let pretrained = wm.load_pretrained(model.to_owned()).await?;
    let lg = pretrained.read().await;
    let config = lg.config();
    let vital = config.normalized_vital()?;
    Ok(web::Json(vital))
}

#[get("/expert/{model}/{layer}/{expert}")]
async fn load_expert(
    req: HttpRequest,
    wm: web::Data<&'static WeightManager<'static>>,
) -> EKResult<Vec<u8>> {
    let model = req.match_info().get("model").unwrap();
    let layer = req.match_info().get("layer").unwrap().parse::<usize>()?;
    let expert = req.match_info().get("expert").unwrap().parse::<usize>()?;
    let pretrained = wm.load_pretrained(model.to_owned()).await?;
    let tv = pretrained.read().await.get_expert(layer, expert).await?;
    Ok(tv)
}

#[get("/weight/{model}/{key}")]
async fn load_layer(
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

pub async fn listen<A: ToSocketAddrs>(roots: &'static [PathBuf], addr: A) -> EKResult<()> {
    let wm = load_manager(roots).await;
    let addr = addr.to_socket_addrs().unwrap().collect::<Vec<_>>();
    log::info!("starting weight server.");
    for a in addr.iter() {
        log::info!("listening on {}", a);
    }
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(wm))
            .service(load_layer)
            .service(load_expert)
            .service(load_meta_vital)
    })
    .bind(addr.as_slice())?
    .run()
    .await
    .map_err(EKError::from)
}

#[cfg(test)]
mod test {
    use std::mem::transmute;

    use crate::safetensor::transformer::VitalMeta;

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
            test::init_service(App::new().app_data(web::Data::new(wm)).service(load_layer)).await;
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

    #[actix_web::test]
    async fn test_load_expert() {
        let root = workspace_root();
        let test_model: PathBuf = root.join("ek-db").join("resources").join("ds-tiny");
        let tm = vec![test_model.clone()];
        let tm: &'static [PathBuf] = unsafe { transmute(tm.as_slice()) };
        let wm = load_manager(tm).await;
        let app =
            test::init_service(App::new().app_data(web::Data::new(wm)).service(load_expert)).await;
        let req = test::TestRequest::default()
            .uri("/expert/ds-tiny/18/32")
            .insert_header(ContentType::plaintext())
            .to_request();
        let resp = test::call_service(&app, req).await;
        let success = resp.status().is_success();
        assert!(success);
        let body = resp.into_body();
        let res = to_bytes(body).await.unwrap();
        let st = safetensors::SafeTensors::deserialize(&res).unwrap();

        let names = st.names();
        assert_eq!(names.len(), 3);
        let expected = vec![
            "model.layers.18.mlp.experts.32.gate_proj.weight",
            "model.layers.18.mlp.experts.32.down_proj.weight",
            "model.layers.18.mlp.experts.32.up_proj.weight",
        ];

        for name in expected {
            assert!(names.contains(&&name.to_string()));
        }
        let tensor = st
            .tensor("model.layers.18.mlp.experts.32.down_proj.weight")
            .unwrap();
        assert_eq!(tensor.shape(), &[16, 8]);
    }

    #[actix_web::test]
    async fn test_load_meta_vital() {
        let root = workspace_root();
        let test_model: PathBuf = root.join("ek-db").join("resources").join("ds-tiny");
        let tm = vec![test_model.clone()];
        let tm: &'static [PathBuf] = unsafe { transmute(tm.as_slice()) };
        let wm = load_manager(tm).await;
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(wm))
                .service(load_meta_vital),
        )
        .await;
        let req = test::TestRequest::default()
            .uri("/meta/vital/ds-tiny")
            .insert_header(ContentType::plaintext())
            .to_request();
        let resp = test::call_service(&app, req).await;
        let success = resp.status().is_success();
        assert!(success);
        let body = resp.into_body();
        let bytes = to_bytes(body).await.unwrap();
        let vital: VitalMeta = serde_json::from_slice(bytes.as_ref()).unwrap();
        assert_eq!(vital.routed_experts, 256);
        assert_eq!(vital.moe_layers, (3, 61));
        assert_eq!(vital.hidden_dim, 16);
        assert_eq!(vital.inter_dim, 8);
    }
}
