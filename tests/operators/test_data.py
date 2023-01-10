"""
Unittest model to test airflow_provider_azure_machinelearning.operators.data

Requires test unittest from the Python Libraries

Run test:
    pytest tests/operators/test_data.py
or,
    python3 -m unittest tests.operators.test_data
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from airflow_provider_azure_machinelearning.operators.machine_learning.data import (
    AzureMachineLearningCreateDataOperator,
    AzureMachineLearningHook,
)


class TestAzureMachineLearningCreateDataOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Data")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_success(self, mock_client, mock_hook, mock_data, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningCreateDataOperator(
            task_id="test_task_id",
            data_asset=mock_data,
            conn_id="test_connection_id",
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.data.create_or_update.assert_called_once_with(mock_data)


if __name__ == "__main__":
    unittest.main()
