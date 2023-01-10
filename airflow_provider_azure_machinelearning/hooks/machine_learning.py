from __future__ import annotations

from typing import Any

from airflow.hooks.base import BaseHook
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential


class AzureMachineLearningHook(BaseHook):
    """
    A hook to interact with Azure Machine Learning.

    :param azure_machine_learning_conn_id
    """

    conn_type: str = "azure_machine_learning"
    conn_name_attr: str = "azure_machine_learning_conn_id"
    hook_name: str = "Azure Machine Learning"

    @staticmethod
    def get_connection_form_widgets() -> dict[str, Any]:
        """Returns connection widgets to add to connection form"""
        from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import StringField

        return {
            "extra__azure_machine_learning__tenantId": StringField(
                lazy_gettext("Tenant ID"), widget=BS3TextFieldWidget()
            ),
            "extra__azure_machine_learning__subscriptionId": StringField(
                lazy_gettext("Subscription ID"), widget=BS3TextFieldWidget()
            ),
            "extra__azure_machine_learning__resource_group_name": StringField(
                lazy_gettext("Resource Group Name"), widget=BS3TextFieldWidget()
            ),
            "extra__azure_machine_learning__workspace_name": StringField(
                lazy_gettext("Workspace Name"), widget=BS3TextFieldWidget()
            ),
        }

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        """Returns custom field behaviour"""
        return {
            "hidden_fields": ["schema", "port", "host", "extra"],
            "relabeling": {
                "login": "Client ID",
                "password": "Secret",
            },
        }

    def __init__(self, azure_machine_learning_conn_id: str = None):
        self.ml_client: MLClient = None
        self.conn_id = azure_machine_learning_conn_id
        super().__init__()

    def get_client(self) -> MLClient:
        if self.ml_client is not None:
            return self.ml_Client

        (
            subscription_id,
            resource_group,
            workspace_name,
            tenant_id,
            client_id,
            client_secret,
        ) = self.get_credentials()

        credential = ClientSecretCredential(
            client_id=client_id, client_secret=client_secret, tenant_id=tenant_id
        )

        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
            user_agent=AzureMachineLearningHook.get_package_signature(),
        )
        return self.ml_client

    def get_credentials(self) -> ():
        if self.conn_id is None:
            raise KeyError("Connection ID cannot be None.")

        conn = self.get_connection(self.conn_id)
        try:
            subscription_id = conn.extra_dejson.get("extra__azure_machine_learning__subscriptionId")
        except KeyError:
            raise ValueError("A Subscription ID is required to connect to Azure Machine Learning.")
        try:
            resource_group = conn.extra_dejson.get("extra__azure_machine_learning__resource_group_name")
        except KeyError:
            raise ValueError("A Resource Group Name is required to connect to Azure Machine Learning.")
        try:
            workspace_name = conn.extra_dejson.get("extra__azure_machine_learning__workspace_name")
        except KeyError:
            raise ValueError("A workspace is required to connect to Azure Machine Learning.")
        try:
            tenant_id = conn.extra_dejson.get("extra__azure_machine_learning__tenantId")
        except KeyError:
            raise ValueError("A Tenant ID is required to connect to Azure Machine Learning.")
        if conn.login is None:
            raise ValueError("A Tenant ID is required when authenticating with Client ID and Secret.")
        if conn.password is None:
            raise ValueError("A Client Secret is required when authenticating with Client ID and Secret.")

        self.log.info(
            f"subID: {subscription_id}, rg: {resource_group}, ws: {workspace_name}, conn.login: {conn.login}, tenantID: {tenant_id}"
        )

        return (
            subscription_id,
            resource_group,
            workspace_name,
            tenant_id,
            conn.login,
            conn.password,
        )

    @staticmethod
    def get_package_signature() -> str:
        """Returns python package_name:version"""
        from airflow_provider_azure_machinelearning.__init__ import get_package_name, get_package_version

        return f"{get_package_name()}:{get_package_version()}"
