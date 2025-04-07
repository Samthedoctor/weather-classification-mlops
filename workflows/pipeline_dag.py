from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data.move_data import move_data
from src.data.preprocess import preprocess_new_data
from src.data.scale_data import scale_data
from src.models.train import train_model

default_args = {
    "owner": "me",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ml_pipeline_dag",
    default_args=default_args,
    start_date=datetime(2025, 4, 5),
    schedule_interval="@weekly",
    catchup=False,
) as dag:
    move_task = PythonOperator(
        task_id="move_data",
        python_callable=move_data
    )
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_new_data
    )
    scale_task = PythonOperator(
        task_id="scale_data",
        python_callable=scale_data
    )
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
    
    move_task >> preprocess_task >> scale_task >> train_task