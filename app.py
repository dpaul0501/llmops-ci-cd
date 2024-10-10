
from flask import Flask, jsonify, request
from transformers import pipeline
import mlflow

app = Flask(__name__)

# Load the fine-tuned model
model = pipeline("text-classification", model="./results")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    prediction = model(text)
    return jsonify(prediction)

@app.route("/metrics", methods=["GET"])
def get_metrics():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fine-tuning-small")
    runs = client.search_runs(experiment.experiment_id)
    metrics = [{"run_id": run.info.run_id, "accuracy": run.data.metrics['accuracy']} for run in runs]
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
