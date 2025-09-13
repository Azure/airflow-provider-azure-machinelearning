"""
Comprehensive test suite for airflow_provider_azure_machinelearning.hooks.machine_learning

Tests all authentication methods including:
- ClientSecretCredential (service principal)
- DefaultAzureCredential with managed identity
- DefaultAzureCredential with workload identity
- DefaultAzureCredential fallback
- Field parsing and backward compatibility

Run tests:
    pytest tests/hooks/test_machine_learning_new.py
or,
    python3 -m unittest tests.hooks.test_machine_learning_new
"""

from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import patch, MagicMock

from airflow.models.connection import Connection

from airflow_provider_azure_machinelearning.hooks.machine_learning import (
    AzureMachineLearningHook,
)

log = logging.getLogger(__name__)

# Test constants
MOCK_TESTSUB = "test-subscription-id"
MOCK_TESTRG = "test-resource-group"
MOCK_TESTWS = "test-workspace"
MOCK_TESTTENANT = "test-tenant-id"
MOCK_TESTCLIENT = "test-client-id"
MOCK_TESTSECRET = "test-client-secret"
MOCK_MANAGED_IDENTITY_CLIENT_ID = "managed-identity-client-id"
MOCK_WORKLOAD_IDENTITY_TENANT_ID = "workload-identity-tenant-id"


class TestAzureMachineLearningHook(unittest.TestCase):
    """Test cases for AzureMachineLearningHook authentication methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.conn_id = "test_aml_connection"

    @patch.object(AzureMachineLearningHook, "get_connection")
    @patch("airflow_provider_azure_machinelearning.hooks.machine_learning.ClientSecretCredential")
    @patch("airflow_provider_azure_machinelearning.hooks.machine_learning.MLClient")
    def test_client_secret_credential_auth(self, mock_ml_client, mock_client_secret_cred, mock_get_conn):
        """Test ClientSecretCredential authentication (service principal)"""
        # Mock connection with client secret credentials
        mock_connection = Connection(
            conn_id=self.conn_id,
            login=MOCK_TESTCLIENT,
            password=MOCK_TESTSECRET,
            extra=json.dumps({
                "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
                "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
                "extra__azure_machine_learning__tenantId": MOCK_TESTTENANT,
            })
        )
        mock_get_conn.return_value = mock_connection

        # Mock credential instance
        mock_credential_instance = MagicMock()
        mock_client_secret_cred.return_value = mock_credential_instance

        # Initialize hook and get client
        hook = AzureMachineLearningHook(self.conn_id)
        client = hook.get_client()

        # Verify ClientSecretCredential was called with correct parameters
        mock_client_secret_cred.assert_called_once_with(
            tenant_id=MOCK_TESTTENANT,
            client_id=MOCK_TESTCLIENT,
            client_secret=MOCK_TESTSECRET
        )

        # Verify MLClient was called with correct parameters
        mock_ml_client.assert_called_once_with(
            credential=mock_credential_instance,
            subscription_id=MOCK_TESTSUB,
            resource_group_name=MOCK_TESTRG,
            workspace_name=MOCK_TESTWS,
            user_agent=AzureMachineLearningHook.get_package_signature(),
        )

    @patch.object(AzureMachineLearningHook, "get_connection")
    @patch("airflow_provider_azure_machinelearning.hooks.machine_learning.DefaultAzureCredential")
    @patch("airflow_provider_azure_machinelearning.hooks.machine_learning.MLClient")
    def test_managed_identity_workload_identity_auth(self, mock_ml_client, mock_default_cred, mock_get_conn):
        """Test DefaultAzureCredential with managed identity and workload identity"""
        # Mock connection with managed identity + workload identity
        mock_connection = Connection(
            conn_id=self.conn_id,
            login=None,
            password=None,
            extra=json.dumps({
                "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
                "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
                "extra__azure_machine_learning__managed_identity_client_id": MOCK_MANAGED_IDENTITY_CLIENT_ID,
                "extra__azure_machine_learning__workload_identity_tenant_id": MOCK_WORKLOAD_IDENTITY_TENANT_ID,
            })
        )
        mock_get_conn.return_value = mock_connection

        # Mock credential instance
        mock_credential_instance = MagicMock()
        mock_default_cred.return_value = mock_credential_instance

        # Initialize hook and get client
        hook = AzureMachineLearningHook(self.conn_id)
        client = hook.get_client()

        # Verify DefaultAzureCredential was called with managed identity parameters
        mock_default_cred.assert_called_once_with(
            managed_identity_client_id=MOCK_MANAGED_IDENTITY_CLIENT_ID,
            workload_identity_tenant_id=MOCK_WORKLOAD_IDENTITY_TENANT_ID,
            additionally_allowed_tenants=[MOCK_WORKLOAD_IDENTITY_TENANT_ID],
        )

        # Verify MLClient was called with correct parameters
        mock_ml_client.assert_called_once_with(
            credential=mock_credential_instance,
            subscription_id=MOCK_TESTSUB,
            resource_group_name=MOCK_TESTRG,
            workspace_name=MOCK_TESTWS,
            user_agent=AzureMachineLearningHook.get_package_signature(),
        )

    @patch.object(AzureMachineLearningHook, "get_connection")
    @patch("airflow_provider_azure_machinelearning.hooks.machine_learning.DefaultAzureCredential")
    @patch("airflow_provider_azure_machinelearning.hooks.machine_learning.MLClient")
    def test_default_azure_credential_fallback(self, mock_ml_client, mock_default_cred, mock_get_conn):
        """Test DefaultAzureCredential fallback when no specific auth is configured"""
        # Mock connection with minimal configuration
        mock_connection = Connection(
            conn_id=self.conn_id,
            login=None,
            password=None,
            extra=json.dumps({
                "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
                "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
            })
        )
        mock_get_conn.return_value = mock_connection

        # Mock credential instance
        mock_credential_instance = MagicMock()
        mock_default_cred.return_value = mock_credential_instance

        # Initialize hook and get client
        hook = AzureMachineLearningHook(self.conn_id)
        client = hook.get_client()

        # Verify DefaultAzureCredential was called without specific parameters
        mock_default_cred.assert_called_once_with()

        # Verify MLClient was called with correct parameters
        mock_ml_client.assert_called_once_with(
            credential=mock_credential_instance,
            subscription_id=MOCK_TESTSUB,
            resource_group_name=MOCK_TESTRG,
            workspace_name=MOCK_TESTWS,
            user_agent=AzureMachineLearningHook.get_package_signature(),
        )

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_field_parsing_new_format(self, mock_get_conn):
        """Test field parsing with new format (without extra__ prefix)"""
        # Mock connection with new format field names
        mock_connection = Connection(
            conn_id=self.conn_id,
            login=MOCK_TESTCLIENT,
            password=MOCK_TESTSECRET,
            extra=json.dumps({
                "subscriptionId": MOCK_TESTSUB,
                "resource_group_name": MOCK_TESTRG,
                "workspace_name": MOCK_TESTWS,
                "tenantId": MOCK_TESTTENANT,
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        # Test _get_field method directly
        extra = mock_connection.extra_dejson
        
        self.assertEqual(hook._get_field(extra, "subscriptionId"), MOCK_TESTSUB)
        self.assertEqual(hook._get_field(extra, "resource_group_name"), MOCK_TESTRG)
        self.assertEqual(hook._get_field(extra, "workspace_name"), MOCK_TESTWS)
        self.assertEqual(hook._get_field(extra, "tenantId"), MOCK_TESTTENANT)

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_field_parsing_backward_compatibility(self, mock_get_conn):
        """Test field parsing with old format (with extra__ prefix) for backward compatibility"""
        # Mock connection with old format field names
        mock_connection = Connection(
            conn_id=self.conn_id,
            login=MOCK_TESTCLIENT,
            password=MOCK_TESTSECRET,
            extra=json.dumps({
                "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
                "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
                "extra__azure_machine_learning__tenantId": MOCK_TESTTENANT,
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        # Test _get_field method directly
        extra = mock_connection.extra_dejson
        
        self.assertEqual(hook._get_field(extra, "subscriptionId"), MOCK_TESTSUB)
        self.assertEqual(hook._get_field(extra, "resource_group_name"), MOCK_TESTRG)
        self.assertEqual(hook._get_field(extra, "workspace_name"), MOCK_TESTWS)
        self.assertEqual(hook._get_field(extra, "tenantId"), MOCK_TESTTENANT)

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_field_parsing_precedence(self, mock_get_conn):
        """Test that new format takes precedence over old format when both exist"""
        # Mock connection with both new and old format field names
        mock_connection = Connection(
            conn_id=self.conn_id,
            extra=json.dumps({
                "subscriptionId": "new-subscription-id",
                "extra__azure_machine_learning__subscriptionId": "old-subscription-id",
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        # Test _get_field method directly
        extra = mock_connection.extra_dejson
        
        # New format should take precedence
        self.assertEqual(hook._get_field(extra, "subscriptionId"), "new-subscription-id")

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_empty_field_handling(self, mock_get_conn):
        """Test that empty string fields are treated as None"""
        # Mock connection with empty fields
        mock_connection = Connection(
            conn_id=self.conn_id,
            extra=json.dumps({
                "subscriptionId": "",
                "tenantId": "",
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        # Test _get_field method directly
        extra = mock_connection.extra_dejson
        
        self.assertIsNone(hook._get_field(extra, "subscriptionId"))
        self.assertIsNone(hook._get_field(extra, "tenantId"))

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_missing_required_fields_subscription_id(self, mock_get_conn):
        """Test error handling when subscription ID is missing"""
        mock_connection = Connection(
            conn_id=self.conn_id,
            extra=json.dumps({
                "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
                "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        with self.assertRaises(ValueError) as context:
            hook._get_azure_ml_config()
        
        self.assertIn("Subscription ID is required", str(context.exception))

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_missing_required_fields_resource_group(self, mock_get_conn):
        """Test error handling when resource group is missing"""
        mock_connection = Connection(
            conn_id=self.conn_id,
            extra=json.dumps({
                "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        with self.assertRaises(ValueError) as context:
            hook._get_azure_ml_config()
        
        self.assertIn("Resource Group Name is required", str(context.exception))

    @patch.object(AzureMachineLearningHook, "get_connection")
    def test_missing_required_fields_workspace_name(self, mock_get_conn):
        """Test error handling when workspace name is missing"""
        mock_connection = Connection(
            conn_id=self.conn_id,
            extra=json.dumps({
                "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
            })
        )
        mock_get_conn.return_value = mock_connection

        hook = AzureMachineLearningHook(self.conn_id)
        
        with self.assertRaises(ValueError) as context:
            hook._get_azure_ml_config()
        
        self.assertIn("workspace name is required", str(context.exception))

    def test_none_connection_id(self):
        """Test error handling when connection ID is None"""
        hook = AzureMachineLearningHook(None)
        
        with self.assertRaises(KeyError) as context:
            hook._get_credential()
        
        self.assertIn("Connection ID cannot be None", str(context.exception))

    def test_get_package_signature(self):
        """Test package signature generation"""
        signature = AzureMachineLearningHook.get_package_signature()
        
        # Should be in format "package_name/version"
        self.assertIsInstance(signature, str)
        self.assertIn("/", signature)

    def test_ml_client_caching(self):
        """Test that MLClient is cached and reused"""
        with patch.object(AzureMachineLearningHook, "get_connection") as mock_get_conn, \
             patch("airflow_provider_azure_machinelearning.hooks.machine_learning.DefaultAzureCredential") as mock_default_cred, \
             patch("airflow_provider_azure_machinelearning.hooks.machine_learning.MLClient") as mock_ml_client:
            
            # Mock connection
            mock_connection = Connection(
                conn_id=self.conn_id,
                extra=json.dumps({
                    "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
                    "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
                    "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
                })
            )
            mock_get_conn.return_value = mock_connection

            hook = AzureMachineLearningHook(self.conn_id)
            
            # Call get_client twice
            client1 = hook.get_client()
            client2 = hook.get_client()
            
            # MLClient should only be instantiated once due to caching
            mock_ml_client.assert_called_once()
            
            # Both calls should return the same client instance
            self.assertEqual(client1, client2)


if __name__ == "__main__":
    unittest.main()
