from __future__ import annotations

import time
from typing import TYPE_CHECKING, Union
import abc

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
        set_default: bool = False,
        conn_id: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.deployment = deployment
        self.waiting = waiting
        self.set_default = set_default
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

        #Setting by default deployment on the endpoint
        if self.set_default and self.waiting:
                                    
            if isinstance(self.deployment, ManagedOnlineDeployment):
                ml_client = self.ml_client.online_endpoints
            elif isinstance(self.deployment, BatchDeployment):
                ml_client = self.ml_client.batch_endpoints
            else:
                self.log.info(f"Kubernetes endpoint doesn't support set_default option.")
                raise RuntimeError("Kubernetes endpoint doesn't support set_default option.")

            try: 
                endpoint = ml_client.get(self.deployment.endpoint_name)
            except AmlExceptions.ResourceNotFoundError:
                self.log.info(f"Endpoint {self.deployment.endpoint_name} does not exist. Returning.")
                return self.deployment.endpoint_name
            endpoint.defaults.deployment_name = self.deployment.name
            try:
                ml_client.begin_create_or_update(endpoint=endpoint)
            except Exception:
                self.log.info(f"Failed to set by default {self.deployment.name} on {self.deployment.endpoint_name}.")
                raise KeyError(f"Failed to set by default {self.deployment.name} on {self.deployment.endpoint_name}.")

        return self.deployment.endpoint_name


class AzureMachineLearningInvokeEndpointOperator(BaseOperator, abc.ABC):
    """
    Executes Azure ML SDK to invoke an endpoint.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param endpoint: The spec of Azure ML endpoint
    :waiting: When true, the operator blocks till the Azure ML endpoint deployment is completed.

    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        endpoint_name: str,
        inputs: dict = {},
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.endpoint_name = endpoint_name
        self.inputs = inputs
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None
        self.job = None
    

class AzureMachineLearningInvokeBatchEndpointOperator(AzureMachineLearningInvokeEndpointOperator):
    """
    Executes Azure ML SDK to invoke an batch endpoint.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param endpoint: The spec of Azure ML endpoint
    :waiting: When true, the operator blocks till the Azure ML endpoint deployment is completed.

    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        waiting: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.waiting = waiting
        self.hook = None
        self.ml_client = None
        self.job = None

    def execute(self, context: Context) -> None:

        self.log.info(
            f"Executing { __class__.__name__} to deploy Batch Endpoint {self.endpoint_name}."
        )
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()
        
        try:
            self.ml_client.batch_endpoints.get(name=self.endpoint_name)
        except AmlExceptions.ResourceNotFoundError:
            self.log.info(f"Endpoint {self.endpoint_name} does not exist. Returning.")
            return self.endpoint_name

        self.log.info(f"Invoking Endpoint: {self.endpoint_name}.")
        self.job = self.ml_client.batch_endpoints.invoke(endpoint_name=self.endpoint_name, inputs=self.inputs)

        try:
            if self.waiting:
                try:
                    self.ml_client.jobs.stream(name=self.job.name)
                    self.log.info(f"{self.job.name} has finished successfully.")
                except AmlExceptions.ServiceRequestError as e:
                    self.log.info(f"Network issues has been encountered")
                    raise e
        except Exception:
            self.log.info(f"{self.job.name} has finished with errors.")
            raise RuntimeError(f"{self.job.name} has finished with errors.")

    def on_kill(self) -> None:
        """
        
        Stopping Azure ML Job triggered by batch endpoint when canceling from airflow

        """
        
        if self.job and self.job.name and self.waiting:
            try:
                handle = self.ml_client.jobs.begin_cancel(self.job.name)
                handle.wait()
            except AmlExceptions.ServiceResponseError as e:
                self.log.info("Network Issues has been encountered when trying to cancel Job %s.", self.returned_job.name)
                raise e                                
            except: 
                self.log.info("Error when trying to cancel Job %s.", self.returned_job.name)                


        self.log.info("Job %s has been cancelled successfully.", self.returned_job.name)


class AzureMachineLearningInvokeOnlineEndpointOperator(AzureMachineLearningInvokeEndpointOperator):
    """
    Executes Azure ML SDK to invoke an batch endpoint.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param endpoint: The spec of Azure ML endpoint
    :waiting: When true, the operator blocks till the Azure ML endpoint deployment is completed.

    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        request_file: str,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.hook = None
        self.ml_client = None
        self.job = None

    def execute(self, context: Context) -> None:

        self.log.info(
            f"Executing { __class__.__name__} to deploy Online Endpoint {self.endpoint_name}."
        )
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()
        
        try:
            self.ml_client.online_endpoints.get(name=self.endpoint_name)
        except AmlExceptions.ResourceNotFoundError:
            self.log.info(f"Online Endpoint {self.endpoint_name} does not exist. Returning.")
            return self.endpoint_name

        self.log.info(f"Invoking Online Endpoint: {self.endpoint_name}.")
        self.job = self.ml_client.online_endpoints.invoke(endpoint_name=self.endpoint_name, inputs=self.inputs)