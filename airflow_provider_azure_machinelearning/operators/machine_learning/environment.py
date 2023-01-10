from __future__ import annotations

from typing import TYPE_CHECKING

from airflow.models import BaseOperator, BaseOperatorLink
from azure.ai.ml.entities import Environment

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.models.taskinstance import TaskInstanceKey
    from airflow.utils.context import Context


class AzureMachineLearningEnvironmentLink(BaseOperatorLink):
    """Constructs a link to Azure MachineLearning Studio environments page."""

    name = "Azure Machine Learning Studio"

    def get_link(
        self,
        operator: BaseOperator,
        *,
        ti_key: TaskInstanceKey,
    ) -> str:
        return "https://ml.azure.com/environments"


class AzureMachineLearningCreateEnvironmentOperator(BaseOperator):
    """
    Executes Azure ML SDK to create an environment.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param environment: The spec of the Azure ML environment to build. This class is defined in
        Azure ML SDK.

    Stores name (id) of Azure ML environment into xcom.
    Returns name (id) of the Azure ML environment.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    operator_extra_links = [
        AzureMachineLearningEnvironmentLink(),
    ]

    def __init__(
        self,
        *,
        environment: Environment,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.environment = environment
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(
            f"Executing the { __class__.__name__} to create new environment {self.environment.name}."
        )
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.returned_environment = self.ml_client.environments.create_or_update(self.environment)
        except Exception as ex:
            self.log.error("Failed to create a new environment. Error:")
            raise ex

        return self.returned_environment.name
