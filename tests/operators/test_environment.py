"""
Unittest model to test airflow_provider_azure_machinelearning.operators.environment

Requires test unittest from the Python Libraries

Run test:
    pytest tests/operators/test_environment.py
or,
    python3 -m unittest tests.operators.test_environment
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from airflow_provider_azure_machinelearning.operators.machine_learning.environment import (
    AzureMachineLearningCreateEnvironmentOperator,
    AzureMachineLearningHook,
)


class TestAzureMachineLearningCreateEnvironmentOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Environment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_success(self, mock_client, mock_hook, mock_environment, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningCreateEnvironmentOperator(
            task_id="test_task_id",
            environment=mock_environment,
            conn_id="test_connection_id",
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.environments.create_or_update.assert_called_once_with(mock_environment)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Environment")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_fail(self, mock_client, mock_hook, mock_environment, mock_context):
        mock_hook.return_value = mock_client
        mock_client.environments.create_or_update.side_effect = Exception

        with pytest.raises(Exception):
            AzureMachineLearningCreateEnvironmentOperator(
                task_id="test_task_id",
                environment=mock_environment,
                conn_id="test_connection_id",
            ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.environments.create_or_update.assert_called_once_with(mock_environment)


if __name__ == "__main__":
    unittest.main()
