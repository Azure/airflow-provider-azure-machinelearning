FROM apache/airflow:latest
COPY ./airflow_provider_azure_machinelearning*.whl /opt/airflow/
run pip install --no-cache \
    /opt/airflow/airflow_provider_azure_machinelearning*.whl
