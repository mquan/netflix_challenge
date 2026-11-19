//! Read all source pipeline jobs using the pinned upstream parser.
use netflix_prize::pipeline::{Pipeline, resolve_pipeline};
use serde_json::{json, Map};
fn main() {
    let mut pipelines = Map::new();
    for split in ["old", "1", "3", "new"] {
        let path = format!("pipeline-{}.toml", split);
        let pipeline = Pipeline::load(&path).unwrap();
        let mut jobs = Map::new();
        for (name, job) in resolve_pipeline(&pipeline) {
            jobs.insert(name, json!({"inputs":job.inputs,"inputs_from":job.inputs_from,
                "outputs":job.outputs,"command":job.cmd}));
        }
        pipelines.insert(split.to_string(), json!({"pipeline":path,"jobs":jobs}));
    }
    println!("{}", serde_json::to_string_pretty(&pipelines).unwrap());
}
