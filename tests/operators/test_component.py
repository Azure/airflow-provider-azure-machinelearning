"""
Unittest model to test airflow_provider_azure_machinelearning.operators.component

Requires test unittest from the Python Libraries

Run test:
    pytest tests/operators/test_component.py
or,
    python3 -m unittest tests.operators.test_component
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from airflow_provider_azure_machinelearning.operators.machine_learning.component import (
    AzureMachineLearningHook,
    AzureMachineLearningLoadAndRegisterComponentOperator,
)


class TestAzureMachineLearningLoadAndRegisterComponentOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("airflow_provider_azure_machinelearning.operators.machine_learning.component.load_component")
    @patch("azure.ai.ml.entities.Component")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_success(self, mock_client, mock_hook, mock_component, mock_loader, mock_context):
        mock_hook.return_value = mock_client
        mock_loader.return_value = mock_component
        mock_client.components.get.side_effect = Exception
        source = "test_path"

        AzureMachineLearningLoadAndRegisterComponentOperator(
            task_id="test_task_id",
            source=source,
            conn_id="test_connection_id",
        ).execute(mock_context)

        mock_loader.assert_called_once_with(source=source)
        mock_hook.assert_called_once()
        mock_client.components.get.assert_called_once()
        mock_client.components.create_or_update.assert_called_once_with(mock_component)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Component")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_invalid_path(self, mock_client, mock_hook, mock_component, mock_context):
        mock_hook.return_value = mock_client
        source = "invalid_file_path"

        with pytest.raises(
            Exception,
            match=f"No such file or directory: {source}",
        ):
            AzureMachineLearningLoadAndRegisterComponentOperator(
                task_id="test_task_id",
                source=source,
                conn_id="test_connection_id",
            ).execute(mock_context)

        mock_hook.assert_not_called()
        mock_client.components.get.assert_not_called()
        mock_client.components.create_or_update.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch("airflow_provider_azure_machinelearning.operators.machine_learning.component.load_component")
    @patch("azure.ai.ml.entities.Component")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_failed_to_upload(
        self, mock_client, mock_hook, mock_component, mock_loader, mock_context
    ):
        mock_hook.return_value = mock_client
        mock_loader.return_value = mock_component
        mock_client.components.get.side_effect = Exception
        mock_client.components.create_or_update.side_effect = Exception
        source = "test_path"

        with pytest.raises(
            Exception,
        ):
            AzureMachineLearningLoadAndRegisterComponentOperator(
                task_id="test_task_id",
                source=source,
                conn_id="test_connection_id",
            ).execute(mock_context)

        mock_loader.assert_called_once_with(source=source)
        mock_hook.assert_called_once()
        mock_client.components.get.assert_called_once()
        mock_client.components.create_or_update.assert_called_once_with(mock_component)


if __name__ == "__main__":
    unittest.main()
