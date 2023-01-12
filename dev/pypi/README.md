# Introduction
This is an example to install the AzureML Airflow provider package into an Airflow instance from Pypi.
This docker-compose.yaml file is modified based on instructions from [Running Airflow in Docker How to Guide](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html). Please refer to that page for more on bringing Airflow up in docker.

# Pre-requisites
- python 3.7 or above
- docker and docker compose

# Installation
To install, run:

```
make install
```
or
```
docker compose create; docker compose start
```

# Browse Locally
Access http://localhost:8081 after installation.
