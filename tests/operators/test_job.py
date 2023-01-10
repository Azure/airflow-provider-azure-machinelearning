"""
Unittest model to test airflow_provider_azure_machinelearning.operators.job

Requires test unittest, and pytest from the Python Libraries

Run test:
    pytest tests/operators/test_job.py
or,
    python3 -m unittest tests.operators.test_job
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from airflow_provider_azure_machinelearning.operators.machine_learning.job import (
    AzureMachineLearningCreateJobOperator,
    AzureMachineLearningHook,
)


class TestAzureMachineLearningCreateJobOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.command")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute(self, mock_client, mock_hook, mock_command_job, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningCreateJobOperator(
            task_id="test_task_id",
            job=mock_command_job,
            waiting=False,
            conn_id="test_connection_id",
        ).execute(mock_context)
        mock_hook.assert_called_once()
        mock_client.jobs.create_or_update.assert_called_once_with(mock_command_job)
        mock_client.jobs.stream.assert_not_called()

        AzureMachineLearningCreateJobOperator(
            task_id="test_task_id",
            job=mock_command_job,
            waiting=True,
            conn_id="test_connection_id",
        ).execute(mock_context)
        self.assertEqual(mock_hook.call_count, 2)
        self.assertEqual(mock_client.jobs.create_or_update.call_count, 2)
        mock_client.jobs.stream.assert_called_once()

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.command")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_kill(self, mock_client, mock_hook, mock_command_job, mock_context):
        mock_hook.return_value = mock_client

        op = AzureMachineLearningCreateJobOperator(
            task_id="test_task_id",
            job=mock_command_job,
            waiting=True,
            conn_id="test_connection_id",
        )
        op.execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.jobs.create_or_update.assert_called_once_with(mock_command_job)


if __name__ == "__main__":
    unittest.main()
