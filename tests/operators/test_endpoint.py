"""
Unittest model to test airflow_provider_azure_machinelearning.operators.model

Requires test unittest from the Python Libraries

Run test:
    pytest tests/operators/test_endpoint.py
or,
    python3 -m unittest tests.operators.test_endpoint
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pytest
from azure.core import exceptions as AmlExceptions

from airflow_provider_azure_machinelearning.operators.machine_learning.endpoint import (
    AzureMachineLearningCreateEndpointOperator,
    AzureMachineLearningDeleteEndpointOperator,
    AzureMachineLearningDeployEndpointOperator,
    AzureMachineLearningHook,
    BatchEndpoint,
    KubernetesOnlineEndpoint,
    ManagedOnlineEndpoint,
)


class TestAzureMachineLearningCreateEndpointOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_managedonline_success(self, mock_client, mock_hook, mock_context):
        with patch(
            "azure.ai.ml.entities.ManagedOnlineEndpoint", new=ManagedOnlineEndpoint
        ) as mock_endpoint_class:
            endpoint_instance = Mock(spec=mock_endpoint_class)
            endpoint_instance.name = "testing mock endpoint"
            self.assertTrue(isinstance(endpoint_instance, (int, mock_endpoint_class)))

            mock_hook.return_value = mock_client
            mock_client.online_endpoints.get.side_effect = AmlExceptions.ResourceNotFoundError

            AzureMachineLearningCreateEndpointOperator(
                task_id="test_task_id",
                endpoint=endpoint_instance,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

            mock_hook.assert_called_once()
            mock_client.online_endpoints.get.assert_called_once()
            mock_client.online_endpoints.begin_create_or_update.assert_called_once_with(
                endpoint=endpoint_instance
            )
            mock_client.batch_endpoints.get.assert_not_called()
            mock_client.batch_endpoints.begin_create_or_update.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_k8_success(self, mock_client, mock_hook, mock_context):
        with patch(
            "azure.ai.ml.entities.ManagedOnlineEndpoint", new=KubernetesOnlineEndpoint
        ) as mock_endpoint_class:
            endpoint_instance = Mock(spec=mock_endpoint_class)
            endpoint_instance.name = "testing mock endpoint"
            self.assertTrue(isinstance(endpoint_instance, (int, mock_endpoint_class)))

            mock_hook.return_value = mock_client
            mock_client.online_endpoints.get.side_effect = AmlExceptions.ResourceNotFoundError

            AzureMachineLearningCreateEndpointOperator(
                task_id="test_task_id",
                endpoint=endpoint_instance,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

            mock_hook.assert_called_once()
            mock_client.online_endpoints.get.assert_called_once()
            mock_client.online_endpoints.begin_create_or_update.assert_called_once_with(
                endpoint=endpoint_instance
            )
            mock_client.batch_endpoints.get.assert_not_called()
            mock_client.batch_endpoints.begin_create_or_update.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_batch_success(self, mock_client, mock_hook, mock_context):
        with patch("azure.ai.ml.entities.ManagedOnlineEndpoint", new=BatchEndpoint) as mock_endpoint_class:
            endpoint_instance = Mock(spec=mock_endpoint_class)
            endpoint_instance.name = "testing mock endpoint"
            self.assertTrue(isinstance(endpoint_instance, (int, mock_endpoint_class)))

            mock_hook.return_value = mock_client
            mock_client.batch_endpoints.get.side_effect = AmlExceptions.ResourceNotFoundError

            AzureMachineLearningCreateEndpointOperator(
                task_id="test_task_id",
                endpoint=endpoint_instance,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

            mock_hook.assert_called_once()
            mock_client.batch_endpoints.get.assert_called_once()
            mock_client.batch_endpoints.begin_create_or_update.assert_called_once_with(
                endpoint=endpoint_instance
            )
            mock_client.online_endpoints.get.assert_not_called()
            mock_client.online_endpoints.begin_create_or_update.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_managedonline_exist(self, mock_client, mock_hook, mock_context):
        with patch(
            "azure.ai.ml.entities.ManagedOnlineEndpoint", new=ManagedOnlineEndpoint
        ) as mock_endpoint_class:
            endpoint_instance = Mock(spec=mock_endpoint_class)
            endpoint_instance.name = "testing mock endpoint"
            self.assertTrue(isinstance(endpoint_instance, (int, mock_endpoint_class)))

            mock_hook.return_value = mock_client

            AzureMachineLearningCreateEndpointOperator(
                task_id="test_task_id",
                endpoint=endpoint_instance,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

            mock_hook.assert_called_once()
            mock_client.online_endpoints.get.assert_called_once()
            mock_client.online_endpoints.begin_create_or_update.assert_not_called()
            mock_client.batch_endpoints.get.assert_not_called()
            mock_client.batch_endpoints.begin_create_or_update.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_k8_exist(self, mock_client, mock_hook, mock_context):
        with patch(
            "azure.ai.ml.entities.ManagedOnlineEndpoint", new=KubernetesOnlineEndpoint
        ) as mock_endpoint_class:
            endpoint_instance = Mock(spec=mock_endpoint_class)
            endpoint_instance.name = "testing mock endpoint"
            self.assertTrue(isinstance(endpoint_instance, (int, mock_endpoint_class)))

            mock_hook.return_value = mock_client

            AzureMachineLearningCreateEndpointOperator(
                task_id="test_task_id",
                endpoint=endpoint_instance,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

            mock_hook.assert_called_once()
            mock_client.online_endpoints.get.assert_called_once()
            mock_client.online_endpoints.begin_create_or_update.assert_not_called()
            mock_client.batch_endpoints.get.assert_not_called()
            mock_client.batch_endpoints.begin_create_or_update.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_batch_exist(self, mock_client, mock_hook, mock_context):
        with patch("azure.ai.ml.entities.ManagedOnlineEndpoint", new=BatchEndpoint) as mock_endpoint_class:
            endpoint_instance = Mock(spec=mock_endpoint_class)
            endpoint_instance.name = "testing mock endpoint"
            self.assertTrue(isinstance(endpoint_instance, (int, mock_endpoint_class)))

            mock_hook.return_value = mock_client

            AzureMachineLearningCreateEndpointOperator(
                task_id="test_task_id",
                endpoint=endpoint_instance,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

            mock_hook.assert_called_once()
            mock_client.batch_endpoints.get.assert_called_once()
            mock_client.batch_endpoints.begin_create_or_update.assert_not_called()
            mock_client.online_endpoints.get.assert_not_called()
            mock_client.online_endpoints.begin_create_or_update.assert_not_called()


class TestAzureMachineLearningDeleteEndpointOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_online_not_exist(self, mock_client, mock_hook, mock_context):
        mock_hook.return_value = mock_client
        mock_client.online_endpoints.get.side_effect = AmlExceptions.ResourceNotFoundError

        endpoint_name = "name"
        AzureMachineLearningDeleteEndpointOperator(
            task_id="test_task_id",
            endpoint_name=endpoint_name,
            endpoint_type="online",
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.online_endpoints.get.assert_called_once()
        mock_client.online_endpoints.begin_delete.assert_not_called()
        mock_client.batch_endpoints.get.assert_not_called()
        mock_client.batch_endpoints.begin_delete.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_batch_not_exist(self, mock_client, mock_hook, mock_context):
        mock_hook.return_value = mock_client
        mock_client.batch_endpoints.get.side_effect = AmlExceptions.ResourceNotFoundError

        endpoint_name = "name"
        AzureMachineLearningDeleteEndpointOperator(
            task_id="test_task_id",
            endpoint_name=endpoint_name,
            endpoint_type="batch",
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.batch_endpoints.get.assert_called_once()
        mock_client.batch_endpoints.begin_delete.assert_not_called()
        mock_client.online_endpoints.get.assert_not_called()
        mock_client.online_endpoints.begin_delete.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_online_exist(self, mock_client, mock_hook, mock_context):
        mock_hook.return_value = mock_client

        endpoint_name = "name"
        AzureMachineLearningDeleteEndpointOperator(
            task_id="test_task_id",
            endpoint_name=endpoint_name,
            endpoint_type="online",
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.online_endpoints.get.assert_called_once()
        mock_client.online_endpoints.begin_delete.assert_called_once_with(name=endpoint_name)
        mock_client.batch_endpoints.get.assert_not_called()
        mock_client.batch_endpoints.begin_delete.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_batch_exist(self, mock_client, mock_hook, mock_context):
        mock_hook.return_value = mock_client

        endpoint_name = "name"
        AzureMachineLearningDeleteEndpointOperator(
            task_id="test_task_id",
            endpoint_name=endpoint_name,
            endpoint_type="batch",
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.batch_endpoints.get.assert_called_once()
        mock_client.batch_endpoints.begin_delete.assert_called_once_with(name=endpoint_name)
        mock_client.online_endpoints.get.assert_not_called()
        mock_client.online_endpoints.begin_delete.assert_not_called()


class TestAzureMachineLearningDeployEndpointOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.ManagedOnlineDeployment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_managedonline_success(self, mock_client, mock_hook, mock_deployment, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningDeployEndpointOperator(
            task_id="test_task_id",
            deployment=mock_deployment,
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.begin_create_or_update.assert_called_once_with(mock_deployment)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.KubernetesOnlineDeployment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_k8_success(self, mock_client, mock_hook, mock_deployment, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningDeployEndpointOperator(
            task_id="test_task_id",
            deployment=mock_deployment,
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.begin_create_or_update.assert_called_once_with(mock_deployment)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.BatchDeployment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_batch_success(self, mock_client, mock_hook, mock_deployment, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningDeployEndpointOperator(
            task_id="test_task_id",
            deployment=mock_deployment,
            conn_id="test_connection_id",
            waiting=False,
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.begin_create_or_update.assert_called_once_with(mock_deployment)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.ManagedOnlineDeployment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_managedonline_fail(self, mock_client, mock_hook, mock_deployment, mock_context):
        mock_hook.return_value = mock_client
        mock_client.begin_create_or_update.side_effect = Exception

        with pytest.raises(
            Exception,
        ):
            AzureMachineLearningDeployEndpointOperator(
                task_id="test_task_id",
                deployment=mock_deployment,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.begin_create_or_update.assert_called_once_with(mock_deployment)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.KubernetesOnlineDeployment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_k8_fail(self, mock_client, mock_hook, mock_deployment, mock_context):
        mock_hook.return_value = mock_client
        mock_client.begin_create_or_update.side_effect = Exception

        with pytest.raises(
            Exception,
        ):
            AzureMachineLearningDeployEndpointOperator(
                task_id="test_task_id",
                deployment=mock_deployment,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.begin_create_or_update.assert_called_once_with(mock_deployment)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.BatchDeployment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_batch_fail(self, mock_client, mock_hook, mock_deployment, mock_context):
        mock_hook.return_value = mock_client
        mock_client.begin_create_or_update.side_effect = Exception

        with pytest.raises(
            Exception,
        ):
            AzureMachineLearningDeployEndpointOperator(
                task_id="test_task_id",
                deployment=mock_deployment,
                conn_id="test_connection_id",
                waiting=False,
            ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.begin_create_or_update.assert_called_once_with(mock_deployment)


if __name__ == "__main__":
    unittest.main()
