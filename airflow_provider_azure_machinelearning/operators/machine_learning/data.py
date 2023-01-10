from __future__ import annotations

from typing import TYPE_CHECKING

from airflow.models import BaseOperator
from azure.ai.ml.entities import Data

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.utils.context import Context


class AzureMachineLearningCreateDataOperator(BaseOperator):
    """
    Executes Azure ML SDK to create a dataset.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param model: The spec of the Azure ML Data to register.

    Returns path of the Azure ML data.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        data_asset: Data,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.data_asset = data_asset
        self.conn_id = conn_id
        self.uploaded_data_asset = None
        self.ml_client = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing the { __class__.__name__}  to create data asset {self.data_asset.name}.")
        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()

        try:
            self.uploaded_data_asset = self.ml_client.data.create_or_update(self.data_asset)

            self.log.info(f"data details: {self.uploaded_data_asset}")
            self.log.info(f"data blob storage location: {self.uploaded_data_asset.path}")
        except Exception as ex:
            self.log.error("Failed to create data asset.")
            raise ex

        return self.uploaded_data_asset.path
