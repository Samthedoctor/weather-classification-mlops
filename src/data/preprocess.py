from utils.spark_utils import get_spark_session
from utils.s3_utils import download_file, file_exists
from utils.logging import logger
from utils.config import BUCKET, INPUT_PREFIX, NON_SCALED_PREFIX

def preprocess_new_data():
    """Preprocess new data from Bucket A.csv, drop id and day, overwrite or append to unscaled output."""
    spark = get_spark_session()
    local_input = "temp_input.csv"
    download_file(BUCKET, INPUT_PREFIX, local_input)
    df_input = spark.read.csv(local_input, header=True)
    
    mode = "overwrite" if not file_exists(BUCKET, f"{NON_SCALED_PREFIX}output.csv") else "append"
    if mode == "append":
        local_non_scaled = "temp_non_scaled.csv"
        download_file(BUCKET, f"{NON_SCALED_PREFIX}output.csv", local_non_scaled)
        df_old = spark.read.csv(local_non_scaled, header=True)
        old_count = df_old.count()
        new_count = df_input.count()
        if new_count <= old_count:
            logger.info("No new data to preprocess")
            return
        df_new = df_input.limit(new_count).subtract(df_input.limit(old_count))
    else:
        df_new = df_input
    
    # Drop id and day, keep original column names
    df_processed = df_new.drop("id", "day").dropna()
    
    df_processed.write.csv(f"s3://{BUCKET}/{NON_SCALED_PREFIX}", mode=mode, header=True)
    logger.info(f"{'Overwrote' if mode == 'overwrite' else 'Appended'} {df_new.count()} rows to unscaled output")