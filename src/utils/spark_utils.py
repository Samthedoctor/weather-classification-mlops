from pyspark.sql import SparkSession
from utils.config import SPARK_MASTER, SPARK_APP_NAME

def get_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .master(SPARK_MASTER) \
        .appName(SPARK_APP_NAME) \
        .getOrCreate()