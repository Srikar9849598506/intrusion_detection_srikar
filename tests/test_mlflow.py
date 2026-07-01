import mlflow
import pandas as pd

def test_mlflow_accuracy_threshold():

    # 1. Point to MLflow tracking store
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # 2. Get experiment
    experiment = mlflow.get_experiment_by_name("XGBoost_Experiment")

    assert experiment is not None, "Experiment not found in MLflow"

    # 3. Get latest run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    assert not runs.empty, "No MLflow runs found"

    # 4. Extract accuracy
    accuracy = runs.iloc[0]["metrics.accuracy"]

    print(f"Latest model accuracy: {accuracy}")

    # 5. TEST CONDITION (THIS IS THE REAL GATE)
    assert accuracy >= 0.90, (
        f"Model accuracy too low: {accuracy:.4f}. Required >= 0.90"
    )