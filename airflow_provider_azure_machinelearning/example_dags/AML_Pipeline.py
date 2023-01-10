"""
This DAG serves as an exmaple to show case how to submit pipeline jobs to Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from azure.ai.ml import Input, load_component
from azure.ai.ml.dsl import pipeline

from airflow_provider_azure_machinelearning.operators.machine_learning.job import (
    AzureMachineLearningCreateJobOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_Pipeline",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(curr_dir, "dags_data/jobs/pipeline/1a_pipeline_with_components_from_yaml/")
    train_model = load_component(source=parent_dir + "/train_model.yml")
    score_data = load_component(source=parent_dir + "/score_data.yml")
    eval_model = load_component(source=parent_dir + "/eval_model.yml")

    @pipeline()
    def pipeline_with_components_from_yaml(
        training_input,
        test_input,
        training_max_epochs=20,
        training_learning_rate=1.8,
        learning_rate_schedule="time-based",
    ):
        """E2E  train-score-eval pipeline with components defined via yaml."""
        # Call component obj as function: apply given inputs & parameters to create a node in pipeline
        train_with_sample_data = train_model(
            training_data=training_input,
            max_epochs=training_max_epochs,
            learning_rate=training_learning_rate,
            learning_rate_schedule=learning_rate_schedule,
        )

        score_with_sample_data = score_data(
            model_input=train_with_sample_data.outputs.model_output,
            test_data=test_input,
        )
        score_with_sample_data.outputs.score_output.mode = "upload"

        eval_with_sample_data = eval_model(scoring_result=score_with_sample_data.outputs.score_output)

        # Return: pipeline outputs
        return {
            "trained_model": train_with_sample_data.outputs.model_output,
            "scored_data": score_with_sample_data.outputs.score_output,
            "evaluation_report": eval_with_sample_data.outputs.eval_output,
        }

    pipeline_job = pipeline_with_components_from_yaml(
        training_input=Input(type="uri_folder", path=parent_dir + "/data/"),
        test_input=Input(type="uri_folder", path=parent_dir + "/data/"),
        training_max_epochs=20,
        training_learning_rate=1.8,
        learning_rate_schedule="time-based",
    )

    # set pipeline level compute
    pipeline_job.settings.default_compute = "simple-cpu-ssh"

    connection_id = "AML_TEST_CONNECTION"
    pipeline_task = AzureMachineLearningCreateJobOperator(
        task_id="pipeline",
        job=pipeline_job,
        waiting=False,
        conn_id=connection_id,
    )

    pipeline_task
