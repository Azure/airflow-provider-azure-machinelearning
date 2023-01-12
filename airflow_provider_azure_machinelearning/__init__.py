__version__ = "0.0.1b1"


def get_package_name():
    return "airflow-provider-azure-machinelearning"


def get_package_version():
    return __version__


def get_provider_info():
    return {
        "package-name": get_package_name(),  # Required
        "name": "Airflow Provider Azure Machine Learning",  # Required
        "description": "Airflow provider package for Azure MachineLearning.",  # Required
        "hook-class-names": [
            "airflow_provider_azure_machinelearning.hooks.machine_learning.AzureMachineLearningHook"
        ],
        "connection-types": [
            {
                "connection-type": "azure_machine_learning",
                "hook-class-name": "airflow_provider_azure_machinelearning.hooks.machine_learning.AzureMachineLearningHook",
            }
        ],
        "extra-links": [
            "airflow_provider_azure_machinelearning.operators.machine_learning.compute.AzureMachineLearningComputeLink",
            "airflow_provider_azure_machinelearning.operators.machine_learning.environment.AzureMachineLearningEnvironmentLink",
            "airflow_provider_azure_machinelearning.operators.machine_learning.job.AzureMachineLearningJobLink",
        ],
        "versions": [__version__],  # Required
    }
