from __future__ import annotations

import time
from typing import TYPE_CHECKING, Union

from airflow.models import BaseOperator, BaseOperatorLink
from azure.ai.ml.entities import AmlCompute, ComputeInstance
from azure.core import exceptions as AmlExceptions

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.models.taskinstance import TaskInstanceKey
    from airflow.utils.context import Context


class AzureMachineLearningComputeLink(BaseOperatorLink):
    """Constructs a link to Azure MachineLearning Studio compute page."""

    name = "Azure Machine Learning Studio"

    def get_link(
        self,
        operator: BaseOperator,
        *,
        ti_key: TaskInstanceKey,
    ) -> str:
        return "https://ml.azure.com/compute/list/training"


class AzureMachineLearningCreateComputeResourceOperator(BaseOperator):
    """
    Executes Azure ML SDK to create a compute cluster or instance.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param compute: The spec of the Azure ML compute resource to create.
        This spec can be a computer cluster, compute instance, etc, according to Azure ML SDK definition.
    :waiting: When true, the operator blocks till the create operation finishes.

    Stores name (id) of the Azure ML compute resource.
    Returns name (id) of the Azure ML compute resource.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    operator_extra_links = [
        AzureMachineLearningComputeLink(),
    ]

    def __init__(
        self,
        *,
        compute: Union[AmlCompute, ComputeInstance],
        conn_id: str = None,
        waiting: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.compute = compute
        self.waiting = waiting
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing the { __class__.__name__}  to create compute target { self.compute.name}.")
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.log.info(f"Checking if cluster {self.compute.name} already exists.")
            self.returned_compute = self.ml_client.compute.get(self.compute.name)
            self.log.info(f"Cluster {self.compute.name} exists")
            self.log.info(f"{self.returned_compute}")
        except AmlExceptions.ResourceNotFoundError:
            self.log.info(f"Cluster {self.compute.name} does not exist. Now creating a new cluster.")
            handle = self.ml_client.compute.begin_create_or_update(self.compute)
            self.log.info(f"Compute resource creation handle: {handle}")
            if self.waiting:
                handle.wait()
                self.log.info(f"Cluster {self.compute.name} created.")
        except Exception:
            self.log.warning("Retrying creating a new cluster.")
            self.returned_compute = self.ml_client.compute.begin_create_or_update(self.compute)

        return self.compute.name


class AzureMachineLearningDeleteComputeResourceOperator(BaseOperator):
    """
    Executes Azure ML SDK to delete a compute cluster or instance.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param compute_name: The name of the Azure ML compute to delete.
    :waiting: When true, the operator blocks till the delete operation finishes.

    Returns name (id) of the Azure ML compute.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    operator_extra_links = [
        AzureMachineLearningComputeLink(),
    ]

    def __init__(
        self,
        *,
        compute_name: str,
        conn_id: str = None,
        waiting: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.compute_name = compute_name
        self.conn_id = conn_id
        self.waiting = waiting
        self.hook = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing the { __class__.__name__}  to delete compute target { self.compute_name}.")
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.log.info(f"Checking if cluster {self.compute_name} already exists.")
            self.returned_compute = self.ml_client.compute.get(self.compute_name)
            self.log.info(f"Cluster {self.compute_name} exists")

            self.log.info(
                f"starting to delete compute resource {self.compute_name} and wait for it the complete."
            )
            handle = self.ml_client.compute.begin_delete(self.compute_name)
            self.log.info(f"Compute resource deletion handle: {handle}")
            if self.waiting:
                handle.wait()
                self.log.info(f"Completed deleting compute resource {self.compute_name}.")
            self.log.info(f"Cluster {self.compute_name} delete request sent.")
        except AmlExceptions.ResourceNotFoundError:
            raise KeyError(f"Cluster {self.compute_name} does not exist. Aborting.")

        return self.compute_name
