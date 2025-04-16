use std::collections::HashMap;

use crate::{
    proto::ek::object::v1::{ExpertSlice, Metadata},
    schema,
};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Queryable, Selectable, Identifiable, PartialEq)]
#[diesel(table_name = schema::node)]
pub struct Node {
    pub id: i32,
    pub hostname: String,
    pub device: String,
    pub config: serde_json::Value,
}

#[derive(
    Serialize,
    Deserialize,
    Clone,
    Debug,
    Queryable,
    Selectable,
    Associations,
    Identifiable,
    PartialEq,
)]
#[diesel(belongs_to(Node))]
#[diesel(table_name = schema::expert)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Expert {
    pub id: i32,
    pub instance_id: i32,
    pub node_id: i32,
    pub expert_id: String,
    pub replica: i32,
    pub state: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug, Queryable, Selectable)]
#[diesel(table_name = schema::instance)]
pub struct Instance {
    pub id: i32,
    pub model_id: i32,
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Queryable, Selectable)]
#[diesel(table_name = schema::model)]
pub struct Model {
    pub id: i32,
    pub name: String,
    pub config: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeWithExperts {
    pub node: Node,
    pub experts: Vec<Expert>,
}

impl From<Vec<Expert>> for ExpertSlice {
    fn from(value: Vec<Expert>) -> Self {
        let expert_meta = value
            .iter()
            .map(|x| Metadata {
                id: x.expert_id.clone(),
                name: x.expert_id.clone(),
                tags: HashMap::new(),
            })
            .collect();
        let slice_meta = Metadata {
            id: "".to_owned(),
            name: "".to_owned(),
            tags: HashMap::new(),
        };
        ExpertSlice {
            meta: Some(slice_meta),
            expert_meta: expert_meta,
        }
    }
}
