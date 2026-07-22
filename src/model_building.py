import os
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# MLflow Setup
# ============================================================

# Use SQLite database as MLflow backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set MLflow experiment
mlflow.set_experiment("XGBoost_Experiment")

# Enable XGBoost autologging
mlflow.xgboost.autolog()


# ============================================================
# Load Data
# ============================================================

def load_data():
    """Load preprocessed data safely."""

    path = "data/preprocessed_data.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The dataset is empty."
        )

    print(
        f"✅ Dataset loaded successfully: {data.shape}"
    )

    return data


# ============================================================
# Build and Train Model
# ============================================================

def build_model():

    # --------------------------------------------------------
    # Load preprocessed data
    # --------------------------------------------------------

    preprocessed_data = load_data()


    # --------------------------------------------------------
    # Create models directory
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )


    # --------------------------------------------------------
    # Separate Features and Target
    # --------------------------------------------------------

    X = preprocessed_data.drop(
        columns=["binary_label"]
    ).astype(float)

    y = preprocessed_data[
        "binary_label"
    ]


    # --------------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Testing samples: {X_test.shape[0]}"
    )


    # --------------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # XGBoost Model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=200,

        max_depth=6,

        learning_rate=0.1,

        random_state=42,

        eval_metric="logloss"
    )


    # ========================================================
    # Start MLflow Run
    # ========================================================

    with mlflow.start_run():

        print(
            "\n🚀 Starting model training..."
        )


        # ----------------------------------------------------
        # Train Model
        # ----------------------------------------------------

        model.fit(
            X_train_scaled,
            y_train
        )


        print(
            "✅ Model training completed"
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        preds = model.predict(
            X_test_scaled
        )


        # ====================================================
        # Calculate Evaluation Metrics
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            preds
        )

        precision = precision_score(
            y_test,
            preds,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            preds,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            preds,
            zero_division=0
        )


        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            preds
        )


        # ====================================================
        # Log Custom Test Metrics to MLflow
        # ====================================================

        mlflow.log_metrics({

            "test_accuracy": accuracy,

            "test_precision": precision,

            "test_recall": recall,

            "test_f1_score": f1

        })


        # ====================================================
        # Save Model
        # ====================================================

        model_path = os.path.join(

            "models",

            "xgb_model.json"

        )


        scaler_path = os.path.join(

            "models",

            "scaler.pkl"

        )


        # Save XGBoost model as JSON
        model.save_model(
            model_path
        )


        # Save scaler
        joblib.dump(

            scaler,

            scaler_path

        )


        # ====================================================
        # Log Artifacts to MLflow
        # ====================================================

        mlflow.log_artifact(

            model_path

        )


        mlflow.log_artifact(

            scaler_path

        )


        # ====================================================
        # Print Results
        # ====================================================

        print(
            "\n======================================"
        )

        print(
            "       MODEL TRAINING RESULTS"
        )

        print(
            "======================================"
        )

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1-Score  : {f1:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            cm
        )

        print(
            "======================================"
        )

        print(
            "✅ Model trained successfully"
        )

        print(
            f"📁 Model saved to: {model_path}"
        )

        print(
            f"📁 Scaler saved to: {scaler_path}"
        )

        print(
            "======================================"
        )


        return model, accuracy


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    build_model()