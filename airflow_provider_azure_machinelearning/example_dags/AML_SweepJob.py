"""
This DAG serves as an exmaple to show case how to submit finetuning (aka sweep) jobs to Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from azure.ai.ml import Input, command
from azure.ai.ml.sweep import Choice, Uniform

from airflow_provider_azure_machinelearning.operators.machine_learning.job import (
    AzureMachineLearningCreateJobOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_SweepJob",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"
    compute_target = "simple-cpu-ssh"

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
        compute=compute_target,
        experiment_name="testing-airflow",
        display_name="iris_training",
    )
    iris_task = AzureMachineLearningCreateJobOperator(
        task_id="iris_training",
        job=iris_command_job,
        waiting=True,
        conn_id=connection_id,
    )

    command_job_for_sweep = iris_command_job(
        learning_rate=Uniform(min_value=0.01, max_value=0.9),
        boosting=Choice(values=["gbdt", "dart"]),
    )
    command_job_for_sweep.experiment_name = "testing-airflow"
    command_job_for_sweep.display_name = "iris_tuning"

    # apply the sweep parameter to obtain the sweep_job
    sweep_job = command_job_for_sweep.sweep(
        compute=compute_target,
        sampling_algorithm="random",
        primary_metric="test-multi_logloss",
        goal="Minimize",
    )

    # define the limits for this sweep
    sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)
    tuning_task = AzureMachineLearningCreateJobOperator(
        task_id="tuning_random_multi_logloss",
        job=sweep_job,
        waiting=True,
        conn_id=connection_id,
    )

    success_task = EmptyOperator(task_id="success")
    iris_task >> tuning_task >> success_task
