from utils.s3_utils import download_file
from utils.logging import logger
from utils.config import BUCKET, NON_SCALED_PREFIX, SCALED_PREFIX, MLFLOW_URI, OPTUNA_TRIALS
from src.models.evaluate import compute_roc_auc
import mlflow
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import pandas as pd
import numpy as np

def load_data(prefix):
    """Load data from S3 and return features and labels."""
    local_file = f"temp_{prefix.split('/')[-2]}.csv"
    download_file(BUCKET, prefix, local_file)
    df = pd.read_csv(local_file)
    X = df.drop("rainfall", axis=1)  # Updated to use 'rainfall' as target
    y = df["rainfall"]
    return X, y

def objective(model_class, X_train, X_val, y_train, y_val, trial):
    """Optuna objective for hyperparameter tuning."""
    if model_class == RandomForestClassifier:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 20)
        }
    elif model_class == DecisionTreeClassifier:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
    elif model_class == xgb.XGBClassifier:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
        }
    elif model_class == SVC:
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "probability": True
        }
    elif model_class == LogisticRegression:
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0),
            "max_iter": 1000
        }
    
    model = model_class(**params)
    model.fit(X_train, y_train)
    train_score = compute_roc_auc(model, X_train, y_train)
    val_score = compute_roc_auc(model, X_val, y_val)
    trial.set_user_attr("train_score", train_score)
    trial.set_user_attr("val_score", val_score)
    return val_score

def train_model():
    """Train multiple models, tune, evaluate, and register the best."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Load and split unscaled data
    X_unscaled, y_unscaled = load_data(f"{NON_SCALED_PREFIX}output.csv")
    X_train_un, X_val_un, y_train_un, y_val_un = train_test_split(X_unscaled, y_unscaled, test_size=0.2, random_state=42)
    
    # Load and split scaled data
    X_scaled, y_scaled = load_data(f"{SCALED_PREFIX}output.csv")
    X_train_sc, X_val_sc, y_train_sc, y_val_sc = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Unscaled models
    unscaled_models = {
        "random_forest": RandomForestClassifier,
        "decision_tree": DecisionTreeClassifier,
        "xgboost": xgb.XGBClassifier
    }
    best_unscaled = {"model": None, "val_score": -1, "train_score": -1, "name": ""}
    
    for name, model_class in unscaled_models.items():
        with mlflow.start_run(run_name=f"unscaled_{name}"):
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(model_class, X_train_un, X_val_un, y_train_un, y_val_un, trial), n_trials=OPTUNA_TRIALS)
            
            best_params = study.best_params
            model = model_class(**best_params)
            model.fit(X_train_un, y_train_un)
            train_score = study.best_trial.user_attrs["train_score"]
            val_score = study.best_trial.user_attrs["val_score"]
            mlflow.log_params(best_params)
            mlflow.log_metric("train_roc_auc", train_score)
            mlflow.log_metric("val_roc_auc", val_score)
            mlflow.sklearn.log_model(model, f"unscaled_{name}")
            logger.info(f"Unscaled {name}: Train ROC-AUC={train_score:.4f}, Val ROC-AUC={val_score:.4f}")
            
            if val_score > best_unscaled["val_score"] and abs(train_score - val_score) < 0.1:
                best_unscaled = {"model": model, "val_score": val_score, "train_score": train_score, "name": name}
    
    # Scaled models
    scaled_models = {
        "svm": SVC,
        "logistic_regression": LogisticRegression
    }
    best_scaled = {"model": None, "val_score": -1, "train_score": -1, "name": ""}
    
    for name, model_class in scaled_models.items():
        with mlflow.start_run(run_name=f"scaled_{name}"):
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(model_class, X_train_sc, X_val_sc, y_train_sc, y_val_sc, trial), n_trials=OPTUNA_TRIALS)
            
            best_params = study.best_params
            model = model_class(**best_params)
            model.fit(X_train_sc, y_train_sc)
            train_score = study.best_trial.user_attrs["train_score"]
            val_score = study.best_trial.user_attrs["val_score"]
            mlflow.log_params(best_params)
            mlflow.log_metric("train_roc_auc", train_score)
            mlflow.log_metric("val_roc_auc", val_score)
            mlflow.sklearn.log_model(model, f"scaled_{name}")
            logger.info(f"Scaled {name}: Train ROC-AUC={train_score:.4f}, Val ROC-AUC={val_score:.4f}")
            
            if val_score > best_scaled["val_score"] and abs(train_score - val_score) < 0.1:
                best_scaled = {"model": model, "val_score": val_score, "train_score": train_score, "name": name}
    
    # Register the best model
    winner = best_scaled if best_scaled["val_score"] > best_unscaled["val_score"] else best_unscaled
    winner_type = "scaled" if winner == best_scaled else "unscaled"
    
    with mlflow.start_run(run_name=f"best_model_{winner['name']}"):
        mlflow.sklearn.log_model(winner["model"], f"best_model_{winner['name']}")
        mlflow.log_metric("train_roc_auc", winner["train_score"])
        mlflow.log_metric("val_roc_auc", winner["val_score"])
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/best_model_{winner['name']}", "BestModel")
        logger.info(f"Registered {winner_type} {winner['name']} with Val ROC-AUC={winner['val_score']:.4f}")