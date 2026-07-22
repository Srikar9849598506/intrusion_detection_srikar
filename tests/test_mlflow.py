import mlflow


def test_mlflow_accuracy_threshold():

    # Connect to MLflow database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Get experiment
    experiment = mlflow.get_experiment_by_name("XGBoost_Experiment")

    assert experiment is not None, "MLflow experiment not found"

    # Get all runs from experiment
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert not runs.empty, "No MLflow runs found"

    # Get latest run
    latest_run = runs.iloc[0]

    # Extract accuracy
    accuracy = latest_run["metrics.accuracy"]

    print(f"Model accuracy: {accuracy}")

    # Accuracy threshold
    threshold = 0.90

    assert accuracy >= threshold, (
        f"Model accuracy {accuracy} is below threshold {threshold}"
    )