"""
This DAG serves as an exmaple to show case how to submit different types of AutoML jobs to Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from azure.ai.ml import Input, automl
from azure.ai.ml.automl import SearchSpace
from azure.ai.ml.constants import AssetTypes, NlpLearningRateScheduler, NlpModels
from azure.ai.ml.sweep import BanditPolicy, Choice, Uniform

from airflow_provider_azure_machinelearning.operators.machine_learning.job import (
    AzureMachineLearningCreateJobOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_AutoML",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    training_mltable_path = os.path.join(
        curr_dir,
        "dags_data/jobs/automl-standalone-jobs/automl-nlp-text-named-entity-recognition-task/training-mltable-folder/",
    )
    validation_mltable_path = os.path.join(
        curr_dir,
        "dags_data/jobs/automl-standalone-jobs/automl-nlp-text-named-entity-recognition-task/validation-mltable-folder/",
    )

    # Training MLTable defined locally, with local data to be uploaded
    my_training_data_input = Input(type=AssetTypes.MLTABLE, path=training_mltable_path)

    # Validation MLTable defined locally, with local data to be uploaded
    my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=validation_mltable_path)

    # general job parameters
    compute_name = "tiny-gpu-ssh"
    exp_name = "testing-airflow"
    exp_timeout = 60

    # Create the AutoML job with the related factory-function.
    text_ner_job = automl.text_ner(
        compute=compute_name,
        # name="dpv2-nlp-text-ner-job-01",
        experiment_name=exp_name,
        display_name="automl_nlp",
        training_data=my_training_data_input,
        validation_data=my_validation_data_input,
        tags={"my_custom_tag": "My custom value"},
    )
    # Set limits
    text_ner_job.set_limits(timeout_minutes=60, max_nodes=2)
    # Pass the fixed parameters
    text_ner_job.set_training_parameters(
        model_name=NlpModels.ROBERTA_BASE,
        learning_rate_scheduler=NlpLearningRateScheduler.LINEAR,
        warmup_ratio=0.1,
    )

    connection_id = "AML_TEST_CONNECTION"
    nlp_task = AzureMachineLearningCreateJobOperator(
        task_id="text_ner_automl_job",
        job=text_ner_job,
        waiting=True,
        conn_id=connection_id,
    )

    text_ner_job_2 = automl.text_ner(
        compute=compute_name,
        experiment_name=exp_name,
        display_name="automl_sweep",
        training_data=my_training_data_input,
        validation_data=my_validation_data_input,
        tags={"my_custom_tag": "My custom value"},
    )
    text_ner_job_2.set_limits(timeout_minutes=120, max_trials=4, max_concurrent_trials=1, max_nodes=4)
    text_ner_job_2.extend_search_space(
        [
            SearchSpace(
                model_name=Choice([NlpModels.BERT_BASE_CASED, NlpModels.ROBERTA_BASE]),
            ),
            SearchSpace(
                model_name=Choice([NlpModels.DISTILROBERTA_BASE]),
                weight_decay=Uniform(0.01, 0.1),
            ),
        ]
    )
    text_ner_job_2.set_sweep(
        sampling_algorithm="Random",
        early_termination=BanditPolicy(evaluation_interval=2, slack_factor=0.05, delay_evaluation=6),
    )
    nlp_tune_task = AzureMachineLearningCreateJobOperator(
        task_id="sweep_text_nerjob",
        job=text_ner_job_2,
        waiting=True,
        conn_id=connection_id,
    )

    # Training MLTable defined locally, with local data to be uploaded
    my_training_data_input = Input(
        type=AssetTypes.MLTABLE,
        path=os.path.join(
            curr_dir,
            "dags_data/jobs/automl-standalone-jobs/automl-forecasting-task-energy-demand/data/training-mltable-folder",
        ),
    )

    # Training MLTable defined locally, with local data to be uploaded
    my_validation_data_input = Input(
        type=AssetTypes.MLTABLE,
        path=os.path.join(
            curr_dir,
            "dags_data/jobs/automl-standalone-jobs/automl-forecasting-task-energy-demand/data/test-mltable-folder",
        ),
    )

    forecasting_job = automl.forecasting(
        compute="simple-cpu-ssh",
        experiment_name=exp_name,
        display_name="automl_forecasting",
        training_data=my_training_data_input,
        target_column_name="demand",
        primary_metric="NormalizedRootMeanSquaredError",
        n_cross_validations=3,
        enable_model_explainability=True,
        tags={"my_custom_tag": "My custom value"},
    )

    # Limits are all optional
    forecasting_job.set_limits(
        timeout_minutes=1200,
        trial_timeout_minutes=40,
        max_trials=10,
        # max_concurrent_trials = 4,
        # max_cores_per_trial: -1,
        enable_early_termination=True,
    )

    # Specialized properties for Time Series Forecasting training
    forecasting_job.set_forecast_settings(
        time_column_name="timeStamp",
        forecast_horizon=48,
        frequency="H",
        target_lags=[12],
        target_rolling_window_size=4,
        # ADDITIONAL FORECASTING TRAINING PARAMS ---
        # time_series_id_column_names=["tid1", "tid2", "tid2"],
        # short_series_handling_config=ShortSeriesHandlingConfiguration.DROP,
        # use_stl="season",
        # seasonality=3,
    )

    # Training properties are optional
    forecasting_job.set_training(blocked_training_algorithms=["ExtremeRandomTrees"])

    forecasting_task = AzureMachineLearningCreateJobOperator(
        task_id="forcasting_automl_job",
        job=forecasting_job,
        waiting=True,
        conn_id=connection_id,
    )

    start_task = EmptyOperator(task_id="start")
    success_task = EmptyOperator(task_id="success")

    start_task >> [nlp_task, nlp_tune_task, forecasting_task] >> success_task
