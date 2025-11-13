"""
NOTE: not used, just an AI draft for exploration!


Wrapper script for creating a REST API for external models.

Idea is that external models can use this in their environment:

pip install chap-core
export TRAIN_CMD="python train.py"
export PREDICT_CMD="python predict.py"
export PORT=8005
chap-runner  # exposes /train, /predict, /jobs/{id}, ...

.. or by importing this:

from chap_core.chap_runner import ModelRunner
import uvicorn

def train_fn(payload, files_dir):
    # payload["training_data"] -> path saved by the runner
    # ... train, save artifacts to files_dir ...
    return {"model_uri": "s3://bucket/modelA:v1", "metrics": {"loss": 0.12}}

def predict_fn(payload, files_dir):
    # may write (files_dir / "out.csv")
    return {"preds_uri": str(files_dir / "out.csv")}

app = ModelRunner(train_fn=train_fn, predict_fn=predict_fn).app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)


"""
