"""
This DAG serves as an exmaple to show case how to upload data assets to Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

from airflow_provider_azure_machinelearning.operators.machine_learning.data import (
    AzureMachineLearningCreateDataOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context

with DAG(
    dag_id="AML_Data",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    code_file_path = os.path.join(
        curr_dir,
        "dags_data/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/data/odFridgeObjects",
    )
    data = Data(
        path=code_file_path,
        type=AssetTypes.URI_FOLDER,
        description="Fridge-items images Object detection",
        name="fridge-items-images-object-detection",
    )
    data_task = AzureMachineLearningCreateDataOperator(
        task_id="upload-data",
        data_asset=data,
        conn_id=connection_id,
    )

    success_task = EmptyOperator(task_id="success")
    data_task >> success_task
