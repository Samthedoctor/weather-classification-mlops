from dotenv import load_dotenv
import os
import yaml

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET = os.getenv("BUCKET")
SOURCE_PREFIX = os.getenv("SOURCE_PREFIX")
INPUT_PREFIX = os.getenv("INPUT_PREFIX")
NON_SCALED_PREFIX = os.getenv("NON_SCALED_PREFIX")
SCALED_PREFIX = os.getenv("SCALED_PREFIX")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
SPARK_MASTER = os.getenv("SPARK_MASTER")
SPARK_APP_NAME = "MyPipeline"

with open("config/pipeline_config.yaml", "r") as f:
    config = yaml.safe_load(f)
BATCH_SIZE = config["data"]["batch_size"]
OPTUNA_TRIALS = config["model"]["optuna_trials"]