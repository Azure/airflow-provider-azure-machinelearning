"""
This DAG serves as an exmaple to show case how to register a componennt in Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG

from airflow_provider_azure_machinelearning.operators.machine_learning.component import (
    AzureMachineLearningLoadAndRegisterComponentOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_Component",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    component_file_path = os.path.join(
        curr_dir,
        "dags_data/assets/component/train.yml",
    )
    AzureMachineLearningLoadAndRegisterComponentOperator(
        task_id="load_and_register_training_component",
        conn_id=connection_id,
        source=component_file_path,
    )
