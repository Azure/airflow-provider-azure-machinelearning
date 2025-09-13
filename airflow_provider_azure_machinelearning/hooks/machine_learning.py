from __future__ import annotations

from typing import Any

from airflow.hooks.base import BaseHook
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import TokenCredential


class AzureMachineLearningHook(BaseHook):
    """
    A hook to interact with Azure Machine Learning.

    :param azure_machine_learning_conn_id
    """

    conn_type: str = "azure_machine_learning"
    conn_name_attr: str = "azure_machine_learning_conn_id"
    hook_name: str = "Azure Machine Learning"

    @classmethod
    def get_connection_form_widgets(cls) -> dict[str, Any]:
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
            "extra__azure_machine_learning__managed_identity_client_id": StringField(
                lazy_gettext("Managed Identity Client ID (optional)"), widget=BS3TextFieldWidget()
            ),
            "extra__azure_machine_learning__workload_identity_tenant_id": StringField(
                lazy_gettext("Workload Identity Tenant ID (optional)"), widget=BS3TextFieldWidget()
            ),
        }

    @classmethod
    def get_ui_field_behaviour(cls) -> dict[str, Any]:
        """Returns custom field behaviour"""
        return {
            "hidden_fields": ["schema", "port", "host", "extra"],
            "relabeling": {
                "login": "Client ID (optional - for ClientSecret auth)",
                "password": "Client Secret (optional - for ClientSecret auth)",
            },
            "placeholders": {
                "login": "Application (client) ID for service principal",
                "password": "Client secret for service principal",
                "extra__azure_machine_learning__tenantId": "Azure tenant ID (required for ClientSecret auth)",
                "extra__azure_machine_learning__subscriptionId": "Azure subscription ID",
                "extra__azure_machine_learning__resource_group_name": "Resource group name",
                "extra__azure_machine_learning__workspace_name": "Azure ML workspace name",
                "extra__azure_machine_learning__managed_identity_client_id": "Client ID for user-assigned managed identity",
                "extra__azure_machine_learning__workload_identity_tenant_id": "Tenant ID for workload identity",
            },
        }

    def __init__(self, azure_machine_learning_conn_id: str | None = None):
        self.ml_client: MLClient | None = None
        self.conn_id = azure_machine_learning_conn_id
        super().__init__()

    def get_client(self) -> MLClient:
        if self.ml_client is not None:
            return self.ml_client

        credential = self._get_credential()
        subscription_id, resource_group, workspace_name = self._get_azure_ml_config()

        self.ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
            user_agent=AzureMachineLearningHook.get_package_signature(),
        )
        return self.ml_client

    def _get_field(self, extra_dict: dict[str, Any], field_name: str) -> str | None:
        """Get field from extra, first checking short name, then for backcompat we check for prefixed name."""
        backcompat_prefix = "extra__azure_machine_learning__"
        backcompat_key = f"{backcompat_prefix}{field_name}"
        
        if field_name.startswith("extra__"):
            raise ValueError(
                f"Got prefixed name {field_name}; please remove the '{backcompat_prefix}' prefix "
                "when using this method."
            )
        
        ret = None
        if field_name in extra_dict:
            if backcompat_key in extra_dict:
                self.log.warning(
                    f"Conflicting params `{field_name}` and `{backcompat_key}` found in extras for conn "
                    f"{self.conn_id}. Using value for `{field_name}`.  Please ensure this "
                    f"is the correct value and remove the backcompat key `{backcompat_key}`."
                )
            ret = extra_dict[field_name]
        elif backcompat_key in extra_dict:
            ret = extra_dict.get(backcompat_key)
            
        if ret == "":
            return None
        return ret

    def _get_credential(self) -> TokenCredential:
        """Get appropriate Azure credential based on connection configuration."""
        if self.conn_id is None:
            raise KeyError("Connection ID cannot be None.")

        conn = self.get_connection(self.conn_id)
        extra = conn.extra_dejson or {}

        # Priority 1: ClientSecretCredential (if tenant_id, client_id, and client_secret are provided)
        tenant_id = self._get_field(extra, "tenantId")
        client_id = conn.login
        client_secret = conn.password

        if tenant_id and client_id and client_secret:
            self.log.info("Using ClientSecretCredential for authentication")
            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )

        # Priority 2: DefaultAzureCredential with managed identity or workload identity
        managed_identity_client_id = self._get_field(extra, "managed_identity_client_id")
        workload_identity_tenant_id = self._get_field(extra, "workload_identity_tenant_id")

        if managed_identity_client_id and workload_identity_tenant_id:
            self.log.info("Using DefaultAzureCredential with managed identity and workload identity")
            return DefaultAzureCredential(
                managed_identity_client_id=managed_identity_client_id,
                workload_identity_tenant_id=workload_identity_tenant_id,
                additionally_allowed_tenants=[workload_identity_tenant_id],
            )

        # Priority 3: DefaultAzureCredential (fallback for managed identity, workload identity, etc.)
        self.log.info("Using DefaultAzureCredential for authentication")
        return DefaultAzureCredential()

    def _get_azure_ml_config(self) -> tuple[str, str, str]:
        """Get Azure ML configuration (subscription_id, resource_group, workspace_name)."""
        if self.conn_id is None:
            raise KeyError("Connection ID cannot be None.")

        conn = self.get_connection(self.conn_id)
        extra = conn.extra_dejson or {}

        subscription_id = self._get_field(extra, "subscriptionId")
        if not subscription_id:
            raise ValueError("A Subscription ID is required to connect to Azure Machine Learning.")

        resource_group = self._get_field(extra, "resource_group_name")
        if not resource_group:
            raise ValueError("A Resource Group Name is required to connect to Azure Machine Learning.")

        workspace_name = self._get_field(extra, "workspace_name")
        if not workspace_name:
            raise ValueError("A workspace name is required to connect to Azure Machine Learning.")

        self.log.info(
            f"Azure ML config: subscription_id={subscription_id}, "
            f"resource_group={resource_group}, workspace_name={workspace_name}"
        )

        return subscription_id, resource_group, workspace_name

    @staticmethod
    def get_package_signature() -> str:
        """Returns python package_name:version"""
        from airflow_provider_azure_machinelearning.__init__ import get_package_name, get_package_version

        return f"{get_package_name()}/{get_package_version()}"
