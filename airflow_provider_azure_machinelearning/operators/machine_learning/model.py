from __future__ import annotations

from typing import TYPE_CHECKING

from airflow.models import BaseOperator
from azure.ai.ml.entities import Model

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.utils.context import Context


class AzureMachineLearningRegisterModelOperator(BaseOperator):
    """
    Executes Azure ML SDK to register a model.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param model: The spec of the Azure ML model to register.


    Returns name (id) of the Azure ML model.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        model: Model,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.model = model
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None
        self.returned_model = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing the { __class__.__name__} to register model {self.model.name}.")
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.returned_model = self.ml_client.models.create_or_update(self.model)
            self.log.info(f"{self.returned_model}")
        except Exception as ex:
            self.log.error("Failed to register model. Error:")
            raise ex
        return self.returned_model.name
