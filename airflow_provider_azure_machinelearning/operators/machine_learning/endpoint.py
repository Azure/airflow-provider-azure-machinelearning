from __future__ import annotations

import time
from typing import TYPE_CHECKING, Union

from airflow.models import BaseOperator
from azure.ai.ml.entities import (
    BatchDeployment,
    BatchEndpoint,
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.core import exceptions as AmlExceptions

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.utils.context import Context


class AzureMachineLearningCreateEndpointOperator(BaseOperator):
    """
    Executes Azure ML SDK to create an Endpoint.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param endpoint: The spec of Azure ML Endpoint
    :waiting: When true, the operator blocks till the Azure ML endpoint is created.

    Returns name(id) of the Azure ML Endpoint.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        endpoint: Union[BatchEndpoint, KubernetesOnlineEndpoint, ManagedOnlineEndpoint],
        waiting: bool = False,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.endpoint = endpoint
        self.waiting = waiting
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing { __class__.__name__} to create new endpoint {self.endpoint.name}.")
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.get_caller().get(name=self.endpoint.name)
            self.log.info(f"Endpoint {self.endpoint.name} already exists. Returning.")
            return self.endpoint.name
        except AmlExceptions.ResourceNotFoundError:
            self.log.info(f"Creating a new Endpoint named {self.endpoint.name}.")
            try:
                self.log.info(f"starting to create Endpoint: {self.endpoint.name}.")
                handle = self.get_caller().begin_create_or_update(endpoint=self.endpoint)
                self.log.info(f"Operation handle: {handle}")
                if self.waiting:
                    handle.wait()
                    self.log.info(f"Completed creating Endpoint: {self.endpoint.name}.")
            except Exception as ex:
                raise KeyError(f"Failed to create a new Endpoint. Error: {ex}")
        return self.endpoint.name

    def get_caller(self):
        if isinstance(self.endpoint, (ManagedOnlineEndpoint, KubernetesOnlineEndpoint)):
            return self.ml_client.online_endpoints
        elif isinstance(self.endpoint, (BatchEndpoint)):
            return self.ml_client.batch_endpoints
        else:
            raise KeyError("Unrecognized Endpoint type.")


class AzureMachineLearningDeleteEndpointOperator(BaseOperator):
    """
    Executes Azure ML SDK to delete an Endpoint.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param endpoint: The spec of Azure ML Endpoint
    :waiting: When true, the operator blocks till the Azure ML endpoint is deleted.

    Returns name(id) of the Azure ML Endpoint.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        endpoint_name: str,
        endpoint_type: str,
        waiting: bool = False,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.endpoint_name = endpoint_name
        self.endpoint_type = endpoint_type
        self.waiting = waiting
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing { __class__.__name__} to Delete Endpoint {self.endpoint_name}.")
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.get_caller().get(name=self.endpoint_name)
        except AmlExceptions.ResourceNotFoundError:
            self.log.info(f"Endpoint {self.endpoint_name} does not exist. Returning.")
            return self.endpoint_name

        self.log.info(f"Deleting an Endpoint: {self.endpoint_name}.")
        try:
            handle = self.get_caller().begin_delete(name=self.endpoint_name)
            self.log.info(f"Operation handle: {handle}")
            if self.waiting:
                handle.wait()
                self.log.info(f"Completed deleting Endpoint: {self.endpoint_name}.")
        except Exception as ex:
            self.log.error("Failed to delete an Endpoint.")
            raise ex

        return self.endpoint_name

    def get_caller(self):
        if self.endpoint_type == "online":
            return self.ml_client.online_endpoints
        elif self.endpoint_type == "batch":
            return self.ml_client.batch_endpoints
        else:
            raise KeyError("Unrecognized Endpoint type.")


class AzureMachineLearningDeployEndpointOperator(BaseOperator):
    """
    Executes Azure ML SDK to deploy a model to an endpoint.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param endpoint: The spec of Azure ML endpoint
    :waiting: When true, the operator blocks till the Azure ML endpoint deployment is completed.

    Returns name(id) of the Azure ML endpoint.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        deployment: Union[BatchDeployment, KubernetesOnlineDeployment, ManagedOnlineDeployment],
        waiting: bool = False,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.deployment = deployment
        self.waiting = waiting
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(
            f"Executing { __class__.__name__} to deploy model to Endpoint {self.deployment.endpoint_name}."
        )
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        self.log.info(f"Deployment started. Endpoint: {self.deployment.endpoint_name}.")
        try:
            handle = self.ml_client.begin_create_or_update(self.deployment)
            self.log.info(f"Operation handle: {handle}")
            if self.waiting:
                handle.wait()
                self.log.info("Completed deployment.")
        except Exception:
            self.log.info(f"Failed to deploy endpoint {self.deployment.endpoint_name}.")
            raise KeyError(f"Failed to deploy endpoint {self.deployment.endpoint_name}.")

        return self.deployment.endpoint_name
