from utils.s3_utils import download_file, upload_file, file_exists
from utils.logging import logger
from utils.config import BUCKET, SOURCE_PREFIX, INPUT_PREFIX, BATCH_SIZE
import pandas as pd

def move_data():
    """Move 150 rows from Bucket source.csv to Bucket A.csv if source exists."""
    local_source = "temp_source.csv"
    if not file_exists(BUCKET, SOURCE_PREFIX):
        logger.info("No source data to move (Week 1)")
        return False
    
    download_file(BUCKET, SOURCE_PREFIX, local_source)
    df_source = pd.read_csv(local_source)
    
    if df_source.empty:
        logger.info("Source file is empty")
        return False
    
    batch = df_source.head(BATCH_SIZE)
    remaining = df_source.iloc[BATCH_SIZE:]
    
    local_input = "temp_input.csv"
    download_file(BUCKET, INPUT_PREFIX, local_input)
    df_input = pd.read_csv(local_input)
    df_input = pd.concat([df_input, batch])
    
    upload_file(df_input.to_csv(index=False), BUCKET, INPUT_PREFIX)
    upload_file(remaining.to_csv(index=False), BUCKET, SOURCE_PREFIX)
    
    logger.info(f"Moved {len(batch)} rows from {SOURCE_PREFIX} to {INPUT_PREFIX}")
    return True