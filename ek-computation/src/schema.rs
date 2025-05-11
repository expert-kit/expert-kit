// @generated automatically by Diesel CLI.

diesel::table! {
    expert (id) {
        id -> Int4,
        instance_id -> Int4,
        node_id -> Int4,
        expert_id -> Text,
        replica -> Int4,
        state -> Jsonb,
    }
}

diesel::table! {
    instance (id) {
        id -> Int4,
        model_id -> Int4,
        name -> Text,
    }
}

diesel::table! {
    model (id) {
        id -> Int4,
        name -> Text,
        config -> Jsonb,
    }
}

diesel::table! {
    node (id) {
        id -> Int4,
        hostname -> Text,
        device -> Text,
        last_seen_at -> Timestamp,
        config -> Jsonb,
        
    }
}

diesel::joinable!(expert -> instance (instance_id));
diesel::joinable!(expert -> node (node_id));
diesel::joinable!(instance -> model (model_id));

diesel::allow_tables_to_appear_in_same_query!(expert, instance, model, node,);
