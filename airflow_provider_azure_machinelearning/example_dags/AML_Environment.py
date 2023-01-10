"""
This DAG serves as an exmaple to show case how to build an custom enviroment/image in Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from azure.ai.ml.entities import Environment

from airflow_provider_azure_machinelearning.operators.machine_learning.environment import (
    AzureMachineLearningCreateEnvironmentOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_Environment",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    conda_file_path = os.path.join(curr_dir, "dags_data/environments/environment_and_train/conda.yml")
    env_1 = Environment(
        name="test_job_env",
        description="Custom environment for Credit Card Defaults pipeline",
        tags={"scikit-learn": "0.24.2"},
        conda_file=conda_file_path,
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
        version="1.1",
    )
    create_environment_task = AzureMachineLearningCreateEnvironmentOperator(
        task_id="build_env_scikit-lear_1dot0",
        conn_id=connection_id,
        environment=env_1,
    )
    create_environment_task
