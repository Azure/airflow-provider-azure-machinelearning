"""
Unittest model to test airflow_provider_azure_machinelearning.operators.model

Requires test unittest from the Python Libraries

Run test:
    pytest tests/operators/test_model.py
or,
    python3 -m unittest tests.operators.test_model
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from airflow_provider_azure_machinelearning.operators.machine_learning.model import (
    AzureMachineLearningHook,
    AzureMachineLearningRegisterModelOperator,
)


class TestAzureMachineLearningCreateEnvironmentOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Model")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_success(self, mock_client, mock_hook, mock_model, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningRegisterModelOperator(
            task_id="test_task_id",
            model=mock_model,
            conn_id="test_connection_id",
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.models.create_or_update.assert_called_once_with(mock_model)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Model")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_failed(self, mock_client, mock_hook, mock_model, mock_context):
        mock_hook.return_value = mock_client
        mock_client.models.create_or_update.side_effect = Exception

        with pytest.raises(
            Exception,
        ):
            AzureMachineLearningRegisterModelOperator(
                task_id="test_task_id",
                model=mock_model,
                conn_id="test_connection_id",
            ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.models.create_or_update.assert_called_once_with(mock_model)


if __name__ == "__main__":
    unittest.main()
