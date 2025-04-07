import boto3
from utils.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_file(local_path, bucket, key):
    """Upload a local file to S3."""
    s3.upload_file(Filename=local_path, Bucket=bucket, Key=key)

def download_file(bucket, key, local_path):
    """Download an S3 file to local."""
    s3.download_file(Bucket=bucket, Key=key, Filename=local_path)

def list_files(bucket, prefix=""):
    """List all files in a bucket with a prefix."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def file_exists(bucket, key):
    """Check if a file exists in S3."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise