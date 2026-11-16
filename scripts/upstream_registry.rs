//! Read the registry and executable DAG through upstream's own parsers.
use netflix_prize::blend::{load_models_toml, select_groups, flatten_groups};
use netflix_prize::pipeline::{Pipeline, resolve_pipeline};
use serde_json::{json, Map, Value};

fn main() {
    let requested: Vec<String> = ["tx", "rbm", "nn", "other", "gbmblend", "mlpblend", "fwlsblend"]
        .iter().map(|s| s.to_string()).collect();
    let model_groups = load_models_toml("models-new.toml");
    let selected = select_groups(&model_groups, &requested);
    let flat = flatten_groups(&selected, &[]);
    let mut groups = Map::new();
    for request in &requested {
        let mut indices: Vec<usize> = Vec::new();
        for leaf in select_groups(&model_groups, &[request.clone()]).keys() {
            if let Some(values) = flat.group_indices.get(leaf) {
                for &i in values {
                    if !indices.contains(&i) { indices.push(i); }
                }
            }
        }
        groups.insert(request.clone(), json!(indices));
    }
    let pipeline = Pipeline::load("pipeline-new.toml").unwrap();
    let mut jobs = Map::new();
    for (name, job) in resolve_pipeline(&pipeline) {
        jobs.insert(name, json!({"inputs":job.inputs,"inputs_from":job.inputs_from,
            "outputs":job.outputs,"command":job.cmd}));
    }
    let output: Value = json!({"names":flat.names,"clip":flat.clip,
                               "groups":groups,"jobs":jobs});
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
