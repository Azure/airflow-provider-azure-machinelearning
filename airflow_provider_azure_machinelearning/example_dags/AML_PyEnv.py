"""
This DAG serves as an exmaple to show case how to submit AML workloads in Python virtual Environments.
This could be handy for cases where engineers favoring different versions of dependency pip packages.
Note that, in the callerables, we are directly submitting the workload to the AzureMachineLearningHook instead of relying on operators.
"""

from __future__ import annotations

import os
import shutil
from datetime import timedelta
from typing import TYPE_CHECKING

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python_operator import PythonVirtualenvOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

"""
This examples shows how one can take advantage of the authention mechanisms built by this package and orchestration provided by Airflow.
One can direclty code on azure-ai-ml SDK interfaces, instead of operators provided by this pckage.
"""
# flake8: noqa: C901
def tain_and_infer_callerable(base_path):

    import logging
    import os

    from azure.ai.ml import Input, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (
        AmlCompute,
        BuildContext,
        Data,
        Environment,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        Model,
    )
    from azure.ai.ml.sweep import Choice, Uniform
    from azure.core import exceptions as AmlExceptions

    from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

    connection_id = "AML_TEST_CONNECTION"
    hook = AzureMachineLearningHook(connection_id)
    ml_client = hook.get_client()

    # Acquire the logger for Azure SDK
    azlogger = logging.getLogger("azure")
    # Set the desired logging level
    azlogger.setLevel(logging.ERROR)

    logging.info("Uploading data asset.")
    try:
        registered_data_asset = ml_client.data.get(name="iris_csv", version="1")
        logging.info("Found data asset. Will not create again")
    except Exception:
        data = Data(
            path=os.path.join(base_path, "jobs/single-step/lightgbm/iris/data/iris.csv"),
            type=AssetTypes.URI_FILE,
            description="Iris Training Data",
            name="iris_csv",
            version="1",
        )
        ml_client.data.create_or_update(data)
        logging.info("Created data asset")
        registered_data_asset = ml_client.data.get(name="iris_csv", version="1")

    logging.info("Creating a training environment")
    job_env = Environment(
        name="af_custom_lightgbm",
        description="Custom environment for lightgbm",
        build=BuildContext(path=os.path.join(base_path, "environments/lightgbm")),
    )
    try:
        ml_client.environments.create_or_update(job_env)
    except Exception as ex:
        logging.error("Failed to create a new environment. Error:")
        raise ex

    logging.info("Creating compute cluster")
    compute_cluster = AmlCompute(
        name="af-pyenv",
        size="Standard_D4s_v3",
        min_instances=0,
        max_instances=4,
    )
    try:
        logging.info(f"Checking if cluster {compute_cluster.name} already exists.")
        returned_compute = ml_client.compute.get(compute_cluster.name)
        logging.info(f"Cluster {compute_cluster.name} exists")
        logging.info(f"{returned_compute}")
    except AmlExceptions.ResourceNotFoundError:
        logging.info(f"Cluster {compute_cluster.name} does not exist. Now creating a new cluster.")
        ml_client.compute.begin_create_or_update(compute_cluster).wait()
        logging.info("Compute resource created")

    logging.info("Training a model.")
    code_file_path = os.path.join(base_path, "jobs/single-step/lightgbm/iris/src")
    iris_command_job = command(
        code=code_file_path,
        command="python main.py --iris-csv ${{inputs.iris_csv}} --learning-rate ${{inputs.learning_rate}} --boosting ${{inputs.boosting}}",
        environment="af_custom_lightgbm@latest",
        inputs={
            "iris_csv": Input(
                type=AssetTypes.URI_FILE,
                path=registered_data_asset.id,
            ),
            "learning_rate": 0.9,
            "boosting": "gbdt",
        },
        compute=compute_cluster.name,
        display_name="lightgbm-iris-example",
        experiment_name="testing-airflow",
        description="iris command job",
    )
    returned_job = ml_client.jobs.create_or_update(iris_command_job)
    job_end_point = returned_job.services["Studio"].endpoint
    logging.info("Waiting for the job to finish.")
    logging.info(f"training job name: {returned_job.name}")
    logging.info(f"training job endpoint: {job_end_point}")
    ml_client.jobs.stream(returned_job.name)

    logging.info("Tunning a model.")
    command_job_for_sweep = iris_command_job(
        learning_rate=Uniform(min_value=0.01, max_value=0.9),
        boosting=Choice(values=["gbdt", "dart"]),
    )
    # apply the sweep parameter to obtain the sweep_job
    sweep_job = command_job_for_sweep.sweep(
        compute=compute_cluster.name,
        sampling_algorithm="random",
        primary_metric="test-multi_logloss",
        goal="Minimize",
    )

    # define the limits for this sweep
    sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)

    # start the sweep job
    returned_tunning_job = ml_client.jobs.create_or_update(sweep_job)
    tunning_job_end_point = returned_tunning_job.services["Studio"].endpoint
    logging.info(f"tunning job full details: {returned_tunning_job}")
    logging.info(f"tunning job name: {returned_tunning_job.name}")
    logging.info(f"tunning job endpoint: {tunning_job_end_point}")
    ml_client.jobs.stream(returned_tunning_job.name)
    job_details = ml_client.jobs.get(returned_tunning_job.name)
    logging.info(f"sweep job full details: {job_details}")
    tunning_winning_job_name = job_details.properties.get("best_child_run_id")
    logging.info(f"sweeping winning job: {tunning_winning_job_name}")

    # delete the compute cluster
    logging.info("Deleting compute cluster.")
    try:
        ml_client.compute.begin_delete(compute_cluster.name)
    except Exception as ex:
        logging.error("Failed to delete compute cluster.")
        raise KeyError(f"Failed to delete compute cluster. {ex}")

    logging.info("Creating an endpoint.")
    endpoint = ManagedOnlineEndpoint(
        name="Airflow-testing-endpoint",
        description="Created from Airflow",
        auth_mode="key",
    )
    try:
        ml_client.online_endpoints.get(name=endpoint.name)
        logging.info(f"ManagedOnlineEndpoint {endpoint.name} already exists. Returning.")
    except AmlExceptions.ResourceNotFoundError:
        logging.info(f"Creating a new Endponit named {endpoint.name}.")
        try:
            logging.info(f"starting to create Endponit: {endpoint.name}.")
            ml_client.online_endpoints.begin_create_or_update(endpoint=endpoint).wait()
            logging.info(f"Completed creating Endponit: {endpoint.name}.")
        except Exception as ex:
            raise KeyError(f"Failed to create a new Endponit. Error: {ex}")

    model = Model(
        path=f"azureml://jobs/{tunning_winning_job_name}/outputs/artifacts/paths/model/",
        name="af-tuned-model",
        description="Model created from tunning.",
        type="mlflow_model",
    )
    logging.info("Registering a model")
    try:
        returned_model = ml_client.models.create_or_update(model)
        logging.info(f"model registered: {returned_model}")
    except Exception as ex:
        logging.error("Failed to register model. Error:")
        raise ex

    blue_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint.name,
        model=model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    logging.info("Deploying model to endpoint.")
    try:
        ml_client.begin_create_or_update(blue_deployment).wait()
        logging.info("Completed deployment.")
    except Exception as ex:
        logging.error(f"Faield to deploy endpoint {blue_deployment.endpoint_name}.")
        raise ex

    # blue deployment takes 100 traffic
    logging.info("Allocating traffic.")
    endpoint.traffic = {blue_deployment.name: 100}
    traffic_allocation = ml_client.begin_create_or_update(endpoint).result()
    logging.info(f"deployment traffic allocation result: {traffic_allocation}")

    logging.info(f"Deleting an Endpoint: {endpoint.name}.")
    try:
        ml_client.online_endpoints.begin_delete(name=endpoint.name).wait()
        logging.info(f"Completed deleting Endpoint: {endpoint.name}.")
    except Exception as ex:
        logging.error("Failed to delete an Endpoint.")
        raise ex


def diabetes_callerable(code_file_path):

    import logging

    from azure.ai.ml import Input, command

    from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

    command_job = command(
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

    connection_id = "AML_TEST_CONNECTION"
    hook = AzureMachineLearningHook(connection_id)
    ml_client = hook.get_client()

    try:
        returned_job = ml_client.jobs.create_or_update(command_job)
        job_end_point = returned_job.services["Studio"].endpoint
        logging.info(f"job endpoint: {job_end_point}")
        logging.info(f"job name: {returned_job.name}")
        logging.info("Waiting for the job to finish.")
        ml_client.jobs.stream(returned_job.name)
    except Exception as ex:
        raise KeyError(f"Failed to submit command job. Error: {ex}")


with DAG(
    dag_id="AML_PythonVirtualenv",
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    schedule_interval="0 18 * * *",
    tags=[],
) as dag:

    if not shutil.which("virtualenv"):
        raise KeyError("The virtalenv_python example task requires virtualenv, please install it.")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(curr_dir, "dags_data/")
    diabetes_code_file_path = os.path.join(
        curr_dir,
        "dags_data/jobs/single-step/scikit-learn/diabetes/src",
    )

    start_task = EmptyOperator(task_id="start")
    success_task = EmptyOperator(task_id="success")

    train_tune_and_infer_task = PythonVirtualenvOperator(
        task_id="train_tune_and_infer",
        python_callable=tain_and_infer_callerable,
        requirements=[
            "azure-ai-ml==1.2.0",
        ],
        op_args=[base_path],
        system_site_packages=True,
        retries=0,
        execution_timeout=timedelta(hours=3),
    )
    start_task >> train_tune_and_infer_task >> success_task

    for aml_release_version in ["1.1.0", "1.1.1"]:
        aml_version_task = EmptyOperator(task_id=f"AML_{aml_release_version}")
        start_task >> aml_version_task
        for blob_version in ["12.14.0b1", "12.14.0b2"]:

            blob_version_task = EmptyOperator(task_id=f"AML_{aml_release_version}_Blob_{blob_version}")

            diabetes_task = PythonVirtualenvOperator(
                task_id=f"AML-{aml_release_version}_Blob-{blob_version}___diabetes_scikit-learn",
                python_callable=diabetes_callerable,
                requirements=[
                    f"azure-ai-ml=={aml_release_version}",
                    f"azure-storage-blob=={blob_version}",
                ],
                op_args=[diabetes_code_file_path],
                system_site_packages=True,
                retries=0,
                execution_timeout=timedelta(minutes=15),
            )

            (aml_version_task >> blob_version_task >> diabetes_task >> success_task)
