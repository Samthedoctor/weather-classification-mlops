from utils.spark_utils import get_spark_session
from utils.logging import logger
from utils.config import BUCKET, NON_SCALED_PREFIX, SCALED_PREFIX

def scale_data():
    """Scale all unscaled features except rainfall, overwrite scaled output."""
    spark = get_spark_session()
    df = spark.read.csv(f"s3://{BUCKET}/{NON_SCALED_PREFIX}*", header=True)
    
    from pyspark.sql.functions import col
    # Define features to scale (exclude rainfall)
    features = ["pressure", "maxtemp", "temperature", "mintemp", 
                "dewpoint", "humidity", "cloud", "sunshine", 
                "winddirection", "windspeed"]
    
    # Compute min and max for each feature
    min_max = {feat: (df.agg({feat: "min"}).collect()[0][0], 
                      df.agg({feat: "max"}).collect()[0][0]) for feat in features}
    
    # Scale features, keep rainfall unscaled
    scaled_expr = [
        ((col(feat) - min_max[feat][0]) / (min_max[feat][1] - min_max[feat][0])).alias(f"scaled_{feat}")
        for feat in features
    ]
    df_scaled = df.select(*scaled_expr, col("rainfall").alias("rainfall"))
    
    df_scaled.write.csv(f"s3://{BUCKET}/{SCALED_PREFIX}", mode="overwrite", header=True)
    logger.info("Scaled data updated in scaled output (rainfall unscaled)")