"""
This DAG serves as an exmaple to show case how to create and to delete an online managed inferencing endpoint in Azure Machine Learning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)

from airflow_provider_azure_machinelearning.operators.machine_learning.compute import (
    AzureMachineLearningCreateComputeResourceOperator,
    AzureMachineLearningDeleteComputeResourceOperator,
)
from airflow_provider_azure_machinelearning.operators.machine_learning.endpoint import (
    AzureMachineLearningCreateEndpointOperator,
    AzureMachineLearningDeleteEndpointOperator,
    AzureMachineLearningDeployEndpointOperator,
)
from airflow_provider_azure_machinelearning.operators.machine_learning.model import (
    AzureMachineLearningRegisterModelOperator,
)

if TYPE_CHECKING:
    from airflow.utils.context import Context


with DAG(
    dag_id="AML_Endpoint",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 17 * * *",
    tags=["AML"],
) as dag:
    connection_id = "AML_TEST_CONNECTION"
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    online_model = Model(
        path="azureml://jobs/{}/outputs/artifacts/paths/model/".format(
            "serene_carpet_4ptn0vn5ty"
        ),  # hard coded for now
        name="run-model-example",
        description="Model created from run.",
        type="mlflow_model",
    )
    register_online_model_task = AzureMachineLearningRegisterModelOperator(
        task_id="register_online_model",
        model=online_model,
        conn_id=connection_id,
    )
    managed_online_endpoint = ManagedOnlineEndpoint(
        name="af-managed-online-endpoint",
        description="Created from Airflow",
        auth_mode="key",
    )
    create_managed_endpoint = AzureMachineLearningCreateEndpointOperator(
        task_id="Create_Online_Managed_Endpoint",
        conn_id=connection_id,
        endpoint=managed_online_endpoint,
        waiting=True,
    )
    managed_deployment = ManagedOnlineDeployment(
        name="ManagedOnline-deployment",
        endpoint_name=managed_online_endpoint.name,
        model=online_model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    deploy_managed_task = AzureMachineLearningDeployEndpointOperator(
        task_id="ManagedOnline_deployment_task",
        conn_id=connection_id,
        deployment=managed_deployment,
        waiting=True,
    )
    delete_managed_endpoint = AzureMachineLearningDeleteEndpointOperator(
        task_id="Delete_Online_Managed_Endpoint",
        conn_id=connection_id,
        endpoint_name=managed_online_endpoint.name,
        endpoint_type="online",
        waiting=True,
    )

    batch_model_name = "heart-classifier"
    batch_model_file_path = os.path.join(
        curr_dir,
        "dags_data/endpints/batch/heart-classifier-mlflow/model",
    )
    batch_model = Model(
        name=batch_model_name,
        path=batch_model_file_path,
        type=AssetTypes.MLFLOW_MODEL,
        description="Airflow Endpoint DAG batch model",
    )
    register_batch_model_task = AzureMachineLearningRegisterModelOperator(
        task_id="register_batch_model",
        model=batch_model,
        conn_id=connection_id,
    )
    batch_endpoint = BatchEndpoint(
        name="af-batch-end-point",
        description="Created from Airflow",
    )
    create_batch_endpoint = AzureMachineLearningCreateEndpointOperator(
        task_id="Create_Batch_Endpoint",
        conn_id=connection_id,
        endpoint=batch_endpoint,
        waiting=True,
    )
    batch_environment = Environment(
        conda_file=os.path.join(
            curr_dir,
            "dags_data/endpints/batch/heart-classifier-mlflow/environment/conda.yml",
        ),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    batch_compute = AmlCompute(
        name="af-test-batch-endpoint",
        size="Standard_DS3_v2",
        min_instances=0,
        max_instances=4,
    )
    batch_compute_create_task = AzureMachineLearningCreateComputeResourceOperator(
        task_id="create_batch_compute_cluster",
        conn_id=connection_id,
        compute=batch_compute,
        waiting=True,
    )
    batch_deployment = BatchDeployment(
        name="airflow-batch-endpoint",
        description="A heart condition classifier based on XGBoost",
        endpoint_name=batch_endpoint.name,
        model=batch_model,
        environment=batch_environment,
        code_configuration=CodeConfiguration(
            code=os.path.join(
                curr_dir,
                "dags_data/endpints/batch/heart-classifier-mlflow/code/",
            ),
            scoring_script="batch_driver_parquet.py",
        ),
        compute=batch_compute.name,
        instance_count=2,
        max_concurrency_per_instance=2,
        mini_batch_size=2,
        output_action=BatchDeploymentOutputAction.SUMMARY_ONLY,
        retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
        logging_level="info",
    )
    deploy_batch_task = AzureMachineLearningDeployEndpointOperator(
        task_id="batch_deployment_task",
        conn_id=connection_id,
        deployment=batch_deployment,
        waiting=True,
    )
    delete_batch_endpoint_task = AzureMachineLearningDeleteEndpointOperator(
        task_id="Delete_Batch_Endpoint",
        conn_id=connection_id,
        endpoint_name=batch_endpoint.name,
        endpoint_type="batch",
        waiting=True,
    )
    delete_batch_compute_task = AzureMachineLearningDeleteComputeResourceOperator(
        task_id="delete_batch_compute_cluster",
        conn_id=connection_id,
        compute_name=batch_compute.name,
    )

    start_online_task = EmptyOperator(task_id="start_online")
    online_success_task = EmptyOperator(task_id="online_success")
    (
        start_online_task
        >> [register_online_model_task, create_managed_endpoint]
        >> deploy_managed_task
        >> delete_managed_endpoint
        >> online_success_task
    )

    start_batch_task = EmptyOperator(task_id="start_batch")
    batch_success_task = EmptyOperator(task_id="batch_success")
    (
        start_batch_task
        >> [register_batch_model_task, create_batch_endpoint, batch_compute_create_task]
        >> deploy_batch_task
        >> [delete_batch_endpoint_task, delete_batch_compute_task]
        >> batch_success_task
    )
