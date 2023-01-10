from __future__ import annotations

from typing import TYPE_CHECKING

from airflow.models import BaseOperator
from azure.ai.ml import load_component

from airflow_provider_azure_machinelearning.hooks.machine_learning import AzureMachineLearningHook

if TYPE_CHECKING:
    from airflow.utils.context import Context


class AzureMachineLearningLoadAndRegisterComponentOperator(BaseOperator):
    """
    Executes Azure ML SDK to register a component.

    :param conn_id: The connection identifier for connecting to Azure Machine Learning.
    :param source: The path to the component.
    """

    ui_color = "white"
    ui_fgcolor = "blue"

    def __init__(
        self,
        *,
        source: str,
        conn_id: str = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        self.source = source
        self.conn_id = conn_id
        self.hook = None
        self.ml_client = None
        self.component = None

    def execute(self, context: Context) -> None:

        self.log.info(f"Executing { __class__.__name__} to create component defined at: {self.source}.")
        try:
            self.component = load_component(source=self.source)
        except Exception as ex:
            self.log.error("Error while loading component at: {self.source}")
            raise ex

        if self.component is None:
            raise KeyError("Failed to prepare a component for registration. Aborting.")

        self.hook = AzureMachineLearningHook(self.conn_id)
        self.ml_client = self.hook.get_client()
        try:
            # try get back the defined component
            self.ml_client.components.get(name=self.component.name, version=self.component.version)
        except Exception:
            self.log.info(
                f"Component {self.component.name} with Version {self.component.version} does not yet exist. Creating it."
            )
            try:
                # create if not exists
                self.ml_client.components.create_or_update(self.component)
                self.log.info(
                    f"Component {self.component.name} with Version {self.component.version} is registered"
                )
            except Exception as ex:
                self.log.error("Error while registering a component:")
                raise ex
