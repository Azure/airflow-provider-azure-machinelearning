"""
Unittest model to test airflow_provider_azure_machinelearning.hooks.machine_learning

Requires test unittest, and pytest from the Python Libraries

Run test:
    pytest tests/hooks/test_machine_learning.py
or,
    python3 -m unittest tests.hooks.test_machine_learning

"""

from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import patch

from airflow.models.connection import Connection

from airflow_provider_azure_machinelearning.hooks.machine_learning import (
    AzureMachineLearningHook,
    ClientSecretCredential,
)

log = logging.getLogger(__name__)

MOCK_TESTSUB = "testsub"
MOCK_TESTRG = "testrg"
MOCK_TESTWS = "testws"
MOCK_TESTTENANT = "testtenant"
MOCK_TESTCLIENT = "testlogin"
MOCK_TESTSECRET = "testsecret"

MOCK_CONNECTION_DB_RECORD = {
    "extra_dejson": {
        "extra__azure_machine_learning__subscriptionId": MOCK_TESTSUB,
        "extra__azure_machine_learning__resource_group_name": MOCK_TESTRG,
        "extra__azure_machine_learning__workspace_name": MOCK_TESTWS,
        "extra__azure_machine_learning__tenantId": MOCK_TESTTENANT,
    },
    "login": MOCK_TESTCLIENT,
    "password": MOCK_TESTSECRET,
}


class TestAzureMachineLearningHook(unittest.TestCase):
    """
    Test functions for AzureMachineLearningHook
    """

    @patch.object(
        AzureMachineLearningHook,
        "get_connection",
    )
    def test_get_credentials(self, mock_get_conn):
        aml_connection = Connection(
            conn_id="test_aml_connection",
            conn_type="Azure Machine Learning",
            login=MOCK_CONNECTION_DB_RECORD.get("login"),
            password=MOCK_CONNECTION_DB_RECORD.get("password"),
            extra=json.dumps(MOCK_CONNECTION_DB_RECORD.get("extra_dejson")),
        )
        mock_get_conn.return_value = aml_connection
        hook = AzureMachineLearningHook("test_connection_id")
        (
            subscription_id,
            resource_group,
            workspace_name,
            tenant_id,
            client_id,
            client_secret,
        ) = hook.get_credentials()
        mock_get_conn.assert_called_once_with("test_connection_id")

        assert subscription_id == MOCK_TESTSUB
        assert resource_group == MOCK_TESTRG
        assert workspace_name == MOCK_TESTWS
        assert tenant_id == MOCK_TESTTENANT
        assert client_id == MOCK_TESTCLIENT
        assert client_secret == MOCK_TESTSECRET
        mock_get_conn.assert_called_once_with("test_connection_id")

    @patch(
        "airflow_provider_azure_machinelearning.hooks.machine_learning.MLClient",
        autospec=True,
    )
    @patch(
        "airflow_provider_azure_machinelearning.hooks.machine_learning.ClientSecretCredential",
        autospec=True,
    )
    @patch.object(
        AzureMachineLearningHook,
        "get_connection",
    )
    def test_get_client(self, mock_get_conn, mock_credential, mock_client):
        aml_connection = Connection(
            conn_id="test_aml_connection",
            conn_type="Azure Machine Learning",
            login=MOCK_CONNECTION_DB_RECORD.get("login"),
            password=MOCK_CONNECTION_DB_RECORD.get("password"),
            extra=json.dumps(MOCK_CONNECTION_DB_RECORD.get("extra_dejson")),
        )
        mock_get_conn.return_value = aml_connection
        mock_credential.return_value = ClientSecretCredential("client", "secret", "tenant")

        hook = AzureMachineLearningHook("test_connection_id")
        hook.get_client()

        mock_get_conn.assert_called_once_with("test_connection_id")
        mock_credential.assert_called_once_with(
            client_id=MOCK_TESTCLIENT,
            client_secret=MOCK_TESTSECRET,
            tenant_id=MOCK_TESTTENANT,
        )
        mock_client.assert_called_once_with(
            credential=mock_credential.return_value,
            subscription_id=MOCK_TESTSUB,
            resource_group_name=MOCK_TESTRG,
            workspace_name=MOCK_TESTWS,
            user_agent=AzureMachineLearningHook.get_package_signature(),
        )

    def test_get_package_signature(self):
        from airflow_provider_azure_machinelearning.__init__ import get_package_name, get_package_version

        assert (
            f"{get_package_name()}/{get_package_version()}"
            == AzureMachineLearningHook.get_package_signature()
        )


if __name__ == "__main__":
    unittest.main()
