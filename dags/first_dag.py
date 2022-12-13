from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

dag =  DAG(
    'tuanvm',
    default_args={
        'email': ['tuanvm@gemvietnam.com'],
        'email_on_failure': True,
        'retries': 1,
        'retry_delay': timedelta(minutes=1),
    },
    description='A simple DAG sample by tuanvm',
    schedule="@once",
    start_date=datetime(2022, 12, 7), # Start date
    tags=['tuanvm'],
)

t1 = BashOperator(
    task_id='print_date',
    bash_command='date > /home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/date.txt',
    dag = dag
)

t2 = BashOperator(
    task_id='sleep',
    bash_command='sleep 5',
    retries=3,
    dag = dag
)
t3 = BashOperator(
    task_id='echo',
    bash_command='echo t3 running',
    dag = dag
)

[t1 , t2] >> t3