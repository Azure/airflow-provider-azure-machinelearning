"""
Unittest model to test airflow_provider_azure_machinelearning.operators.compute

Requires test unittest, and pytest from the Python Libraries

Run test:
    pytest tests/operators/test_compute.py
or,
    python3 -m unittest tests.operators.test_compute
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest
from azure.core import exceptions as AmlExceptions

from airflow_provider_azure_machinelearning.operators.machine_learning.compute import (
    AzureMachineLearningCreateComputeResourceOperator,
    AzureMachineLearningDeleteComputeResourceOperator,
    AzureMachineLearningHook,
)


class TestAzureMachineLearningCreateComputeResourceOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.Compute")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_none_existing(self, mock_client, mock_hook, mock_compute, mock_context):
        mock_hook.return_value = mock_client
        mock_client.compute.get.side_effect = AmlExceptions.ResourceNotFoundError

        AzureMachineLearningCreateComputeResourceOperator(
            task_id="test_task_id",
            compute=mock_compute,
            waiting=False,
            conn_id="test_connection_id",
        ).execute(mock_context)
        mock_hook.assert_called_once()
        mock_client.compute.get.assert_called_once()
        mock_client.compute.begin_create_or_update.assert_called_once_with(mock_compute)

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.AmlCompute")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_existing(self, mock_client, mock_hook, mock_compute, mock_context):
        mock_hook.return_value = mock_client

        AzureMachineLearningCreateComputeResourceOperator(
            task_id="test_task_id",
            compute=mock_compute,
            waiting=False,
            conn_id="test_connection_id",
        ).execute(mock_context)
        mock_hook.assert_called_once()
        mock_client.compute.get.assert_called_once()
        mock_client.compute.begin_create_or_update.assert_not_called()


class TestAzureMachineLearningDeleteComputeResourceOperator(unittest.TestCase):
    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.AmlCompute")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_none_existing(self, mock_client, mock_hook, mock_compute, mock_context):
        mock_hook.return_value = mock_client
        mock_client.compute.get.side_effect = AmlExceptions.ResourceNotFoundError

        compute_name = "test_compute_name"
        with pytest.raises(
            Exception,
            match=f"Cluster {compute_name} does not exist. Aborting.",
        ):
            AzureMachineLearningDeleteComputeResourceOperator(
                task_id="test_task_id",
                compute_name=compute_name,
                waiting=False,
                conn_id="test_connection_id",
            ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.compute.get.assert_called_once()
        mock_client.compute.begin_delete.assert_not_called()

    @patch("airflow.utils.context.Context")
    @patch("azure.ai.ml.entities.ComputeInstance")
    @patch.object(
        AzureMachineLearningHook,
        "get_client",
    )
    @patch("azure.ai.ml.MLClient")
    def test_execute_existing(self, mock_client, mock_hook, mock_compute, mock_context):
        mock_hook.return_value = mock_client

        compute_name = "test_compute_name"
        AzureMachineLearningDeleteComputeResourceOperator(
            task_id="test_task_id",
            compute_name=compute_name,
            waiting=False,
            conn_id="test_connection_id",
        ).execute(mock_context)

        mock_hook.assert_called_once()
        mock_client.compute.get.assert_called_once()
        mock_client.compute.begin_delete.assert_called_once_with(compute_name)


if __name__ == "__main__":
    unittest.main()
