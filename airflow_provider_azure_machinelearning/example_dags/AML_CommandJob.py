"""
This DAG serves as an exmaple to show case how to submit command jobs to Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from azure.ai.ml import Input, command

from airflow_provider_azure_machinelearning.operators.machine_learning.job import (
    AzureMachineLearningCreateJobOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_CommandJob",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    code_file_path = os.path.join(curr_dir, "dags_data/jobs/single-step/lightgbm/iris/src")
    iris_command_job = command(
        code=code_file_path,
        command="python main.py --iris-csv ${{inputs.iris_csv}} --learning-rate ${{inputs.learning_rate}} --boosting ${{inputs.boosting}}",
        environment="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu@latest",
        inputs={
            "iris_csv": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
            ),
            "learning_rate": 0.9,
            "boosting": "gbdt",
        },
        compute="simple-cpu-ssh",
        display_name="lightgbm-iris-example",
        experiment_name="testing-airflow",
        description="iris command job",
    )
    iris_task = AzureMachineLearningCreateJobOperator(
        task_id="iris",
        job=iris_command_job,
        waiting=True,
        conn_id=connection_id,
    )

    code_file_path = os.path.join(curr_dir, "dags_data/jobs/single-step/scikit-learn/diabetes/src")
    diabetes_command_job = command(
        code=code_file_path,
        command="python main.py --diabetes-csv ${{inputs.diabetes}}",
        inputs={
            "diabetes": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv",
            )
        },
        environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
        compute="simple-cpu-ssh",
        display_name="sklearn-diabetes-example",
        experiment_name="testing-airflow",
        description="diabetes command job",
    )
    diabetes_task = AzureMachineLearningCreateJobOperator(
        task_id="diabetes",
        job=diabetes_command_job,
        waiting=False,
        conn_id=connection_id,
    )
    iris_task >> diabetes_task

    success_task = EmptyOperator(task_id="success")
    diabetes_task >> success_task
