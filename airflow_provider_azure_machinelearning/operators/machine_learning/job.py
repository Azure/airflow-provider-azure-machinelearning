from __future__ import annotations

from typing import TYPE_CHECKING, Any

from airflow.models import BaseOperator, BaseOperatorLink
from airflow.models.xcom import XCom

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.models.taskinstance import TaskInstanceKey
    from airflow.utils.context import Context


class AzureMachineLearningJobLink(BaseOperatorLink):
    """Constructs a link to Azure MachineLearning Studio job page."""

    name = "Azure Machine Learning Studio"

    def get_link(
        self,
        operator: BaseOperator,
        *,
        ti_key: TaskInstanceKey,
    ) -> str:
        job_end_point = XCom.get_value(key="job_end_point", ti_key=ti_key)
        if job_end_point:
            return job_end_point
        else:
            return "https://ml.azure.com/runs/"


class AzureMachineLearningCreateJobOperator(BaseOperator):
    """
    Executes Azure ML SDK to create a command job.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param job: The spec of the Azure ML job to execute. This spec can be a Command job, AutoML job,
        Pipeline job, Sweep job, etc, according to Azure ML SDK definition.
    :waiting: When true, the operator blocks till the Azure ML job finishes.

    Stores name (id) and web link to the job on Azure ML Studio into xcom.
    Returns name (id) of the Azure ML job.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    operator_extra_links = [
        AzureMachineLearningJobLink(),
    ]

    def __init__(
        self,
        *,
        conn_id: str = None,
        job: Any = None,
        waiting: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.job = job
        self.conn_id = conn_id
        self.waiting = waiting
        self.returned_job = None
        self.ml_client = None
        self.hook = None
        self.job_end_point = None

    def execute(self, context: Context) -> None:

        self.log.info(
            f"Executing the { __class__.__name__}  to create compute target {self.job.display_name}."
        )
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.returned_job = self.ml_client.jobs.create_or_update(self.job)
            self.job_end_point = self.returned_job.services["Studio"].endpoint
            context["ti"].xcom_push(key="job_id", value=self.returned_job.name)
            context["ti"].xcom_push(key="job_end_point", value=self.job_end_point)
            self.log.info(f"job name: {self.returned_job.name}")
            self.log.info(f"job endpoint: {self.job_end_point}")
        except Exception as ex:
            self.log.error("Failed to submit command job.")
            raise ex

        if self.waiting:
            self.log.info("Waiting for the job to finish.")
            self.ml_client.jobs.stream(self.returned_job.name)

        return self.returned_job.name

    def on_kill(self) -> None:
        if self.returned_job and self.returned_job.name:
            self.ml_client.jobs.cancel(self.returned_job.name)
        self.log.info("Job %s has been cancelled successfully.", self.returned_job.name)
