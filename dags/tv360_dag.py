from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os
import sys
import shutil
import pytz
# sys.path.append("/home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/")
# from TV360.RCM.train import train
from datetime import datetime, timezone, timedelta

timezone_offset = +7.0  # Pacific Standard Time (UTC−08:00)
tzinfo = timezone(timedelta(hours=timezone_offset))

def query_db():
    data_folder = "/home/admin1/mnt_raid/source/tuanvm/Viettel/demographic_data/data/"
    for csv in os.listdir(data_folder):
        shutil.copy("/home/admin1/mnt_raid/source/tuanvm/Viettel/demographic_data/data/" + csv, "data/" + csv)
    return True
    
with DAG(
    'tv360',
    default_args={
        'email': ['tuanvm@gemvietnam.com'],
        'email_on_failure': True,
        'retries': 3,
        'retry_delay': timedelta(minutes=160),
    },
    description='TV360 ML training pipeline DAG',
    schedule='22 08 * * 1-6',
    start_date=datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')) - timedelta(days=2), # Start date
    tags=['tuanvm'],
) as dag:
    
    #Query Database Task
    query_db_task = PythonOperator(task_id='Query-Database', python_callable=query_db, dag=dag)
    
    #Training Task
    
    #Config for Training
    today = datetime.now()
    if today.weekday() == 0:
        train_weights = "best.pt"
    else:
        train_weights = "None"    
    train_epochs = 1
    train_batch_size = 512
    train_device = "cuda:0"
    train_output_dir = "/home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/TV360/RCM/save_models/"
    train_bash_command = "source /home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/dags/run_train.sh \'" + train_weights + "\' " + str(train_epochs) + " " + str(train_batch_size) + " " + str(train_device) + " \'" + train_output_dir + "\'"
    
    training_task = BashOperator(
        task_id='Training-Task',
        bash_command=train_bash_command,
        dag=dag
    )
    
    #Eval Task
    eval_weights = "/home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/TV360/RCM/save_models/85_best_accuracy.pth"
    eval_batch_size = 512
    eval_device = "cuda:0"
    eval_start_day = 20220626
    eval_end_day = 20220630
    eval_bash_command = "source /home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/dags/run_eval.sh \'" + eval_weights + "\' " + str(eval_batch_size) + " \'" + eval_device + "\' " + str(eval_start_day) + " " + str(eval_end_day)
    
    print(eval_bash_command)    
    eval_task = BashOperator(
        task_id='Eval-Task',
        bash_command=eval_bash_command,
        dag=dag
    )
    
    #Inference Task
    inference_weights = "/home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/TV360/RCM/save_models/85_best_accuracy.pth"
    inference_device = 512
    inference_top_k = 100
    
    inference_bash_command = "source /home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/dags/run_eval.sh \'" + inference_weights + "\' " + str(inference_device) + " \' " + inference_top_k
    
    inference_task = BashOperator(
        task_id='Inference-Task',
        bash_command=inference_bash_command,
        dag=dag
    )
    
    #DAGs
    query_db_task >> training_task >> eval_task >> inference_task