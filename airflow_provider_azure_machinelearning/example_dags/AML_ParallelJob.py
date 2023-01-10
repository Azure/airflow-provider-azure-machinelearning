"""
This DAG serves as an exmaple to show case how to submit parallel jobs to Azure Machine Learning.
"""

from __future__ import annotations

import os

import pendulum
from airflow import DAG
from azure.ai.ml import Input, Output, load_component
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment
from azure.ai.ml.parallel import RunFunction, parallel_run_function

from airflow_provider_azure_machinelearning.operators.machine_learning.job import (
    AzureMachineLearningCreateJobOperator,
)

with DAG(
    dag_id="AML_ParallelJob",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(curr_dir, "dags_data/jobs/pipeline/1g_pipeline_with_parallel_nodes/")
    prepare_data = load_component(source=parent_dir + "src/prepare_data.yml")
    code_file_path = os.path.join(parent_dir, "src/")
    conda_file_path = os.path.join(parent_dir, "src/environment_parallel.yml")
    dataset_path = os.path.join(parent_dir, "dataset/")
    model_path = os.path.join(parent_dir, "model/")

    outputs = dict(job_output_path=Output(type=AssetTypes.MLTABLE))

    # parallel task to process file data
    file_batch_inference = parallel_run_function(
        name="file_batch_score",
        display_name="Batch Score with File Dataset",
        description="parallel component for batch score",
        inputs=dict(
            job_data_path=Input(
                type=AssetTypes.MLTABLE,
                description="The data to be split and scored in parallel",
            )
        ),
        outputs=dict(job_output_path=Output(type=AssetTypes.MLTABLE)),
        input_data="${{inputs.job_data_path}}",
        instance_count=2,
        max_concurrency_per_instance=1,
        mini_batch_size="1",
        mini_batch_error_threshold=1,
        retry_settings=dict(max_retries=2, timeout=60),
        logging_level="DEBUG",
        task=RunFunction(
            code=code_file_path,
            entry_script="file_batch_inference.py",
            program_arguments="--job_output_path ${{outputs.job_output_path}}",
            environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
        ),
    )

    # parallel task to process tabular data
    tabular_batch_inference = parallel_run_function(
        name="batch_score_with_tabular_input",
        display_name="Batch Score with Tabular Dataset",
        description="parallel component for batch score",
        inputs=dict(
            job_data_path=Input(
                type=AssetTypes.MLTABLE,
                description="The data to be split and scored in parallel",
            ),
            score_model=Input(type=AssetTypes.URI_FOLDER, description="The model for batch score."),
        ),
        outputs=dict(job_output_path=Output(type=AssetTypes.MLTABLE)),
        input_data="${{inputs.job_data_path}}",
        instance_count=2,
        max_concurrency_per_instance=2,
        mini_batch_size="100",
        mini_batch_error_threshold=5,
        logging_level="DEBUG",
        retry_settings=dict(max_retries=2, timeout=60),
        task=RunFunction(
            code=code_file_path,
            entry_script="tabular_batch_inference.py",
            environment=Environment(
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                conda_file=conda_file_path,
            ),
            program_arguments="--model ${{inputs.score_model}} "
            "--job_output_path ${{outputs.job_output_path}} "
            "--error_threshold 5 "
            "--allowed_failed_percent 30 "
            "--task_overhead_timeout 1200 "
            "--progress_update_timeout 600 "
            "--first_task_creation_timeout 600 "
            "--copy_logs_to_parent True "
            "--resource_monitor_interva 20 ",
            append_row_to="${{outputs.job_output_path}}",
        ),
    )

    @pipeline()
    def parallel_in_pipeline(pipeline_job_data_path, pipeline_score_model):

        prepare_file_tabular_data = prepare_data(input_data=pipeline_job_data_path)
        # output of file & tabular data should be type MLTable
        prepare_file_tabular_data.outputs.file_output_data.type = AssetTypes.MLTABLE
        prepare_file_tabular_data.outputs.tabular_output_data.type = AssetTypes.MLTABLE

        batch_inference_with_file_data = file_batch_inference(
            job_data_path=prepare_file_tabular_data.outputs.file_output_data
        )
        # use eval_mount mode to handle file data
        batch_inference_with_file_data.inputs.job_data_path.mode = InputOutputModes.EVAL_MOUNT
        batch_inference_with_file_data.outputs.job_output_path.type = AssetTypes.MLTABLE

        batch_inference_with_tabular_data = tabular_batch_inference(
            job_data_path=prepare_file_tabular_data.outputs.tabular_output_data,
            score_model=pipeline_score_model,
        )
        # use direct mode to handle tabular data
        batch_inference_with_tabular_data.inputs.job_data_path.mode = InputOutputModes.DIRECT

        return {
            "pipeline_job_out_file": batch_inference_with_file_data.outputs.job_output_path,
            "pipeline_job_out_tabular": batch_inference_with_tabular_data.outputs.job_output_path,
        }

    pipeline_job_data_path = Input(path=dataset_path, type=AssetTypes.MLTABLE, mode=InputOutputModes.RO_MOUNT)
    pipeline_score_model = Input(path=model_path, type=AssetTypes.URI_FOLDER, mode=InputOutputModes.DOWNLOAD)
    # create a pipeline
    pipeline_job = parallel_in_pipeline(
        pipeline_job_data_path=pipeline_job_data_path,
        pipeline_score_model=pipeline_score_model,
    )
    pipeline_job.outputs.pipeline_job_out_tabular.type = AssetTypes.URI_FILE

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
