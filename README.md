<h1 align="center">
Airflow Provider for Azure Machine Learning
</h1>

[Source Code](https://github.com/Azure/airflow-provider-azure-machinelearning) | [Package_PyPI](https://pypi.org/project/airflow-provider-azure-machinelearning/) | [Example DAGs](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/airflow_provider_azure_machinelearning/example_dags) | [Example Docker Containers](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/dev)

This package enables you to submit workflows to Azure Machine Learning from Apache Airflow.

# Pre-requisites

- [Azure Account](https://azure.microsoft.com/en-us/get-started/azure-portal) and [Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning) workspace
    - To verfiy your workspace is set up successfully, you can try to access your workspace at [Azure Machine Learning Studio](https://ml.azure.com/), and try to perform basic actions like allocating compute clusters and submittnig a training job, etc.
- A running [Apache Airflow](https://airflow.apache.org/) instance.

# Installation
In you Apache Airflow instance, run:
```
pip install airflow-provider-azure-machinelearning
```
Or, try it out by following examples in the [dev folder](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/dev/), or Airflow's [How-to-Guide](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) to set up Airflow in Docker containers.

# Configure Azure Machine Learning Connections in Airflow

To send workload to your Azure Machine Learning workspace from Airflow, you need to set up an "Azure Machine Learning" Connection in your Airflow instance:
1. Make sure this package is installed to your Airflow instance. Without this, you will not see "Azure Machine Learning" in the drop down in step 3 and will not be able to add this type of connections.

2. On Airflow web portal, navigate to ```Admin``` --> ```Connections```, and click on ```+``` to add a new connection.

3. From the "Connection Type" dropdown, select "Azure Machine Learning". You should see a form like below
   ![](https://github.com/Azure/airflow-provider-azure-machinelearning/blob/main/resources/Airflow_AzureMachineLearning_Connection.jpg "Azure Machine Learning Connection")

4. ```Connection Id```is a unique identifier for your connection. You will also need to pass this string into AzureML Airflow operators. Check out those [example dags](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/airflow_provider_azure_machinelearning/example_dags/).

5.  ```Description``` is optional. All other fields are required.

6. ```Tenant ID```. You can follow [this instruction](https://learn.microsoft.com/en-us/azure/active-directory/fundamentals/active-directory-how-to-find-tenant "How to find your Azure Active Directory tenant ID") to retrieve it.

7. ```Subscription ID```, ```Resource Group Name```, and ```Workspace Name``` can uniquely identify your workspace in Azure Machine Learning. After opening [Azure Machine Learning Studio](https://ml.azure.com/home), select the desired workspace, then click the "Change workspace" on the upper-right corner of the website (to the left of the profile icon). Here you can find the ```Workspace Name```. Now, click "View All Properties in Azure Portal'. This is Azure resource page of your workspace. From there you can retrieve ```Subscription ID```, and ```Resource Group Name```.

8. ```Client ID``` and ```Secret``` are a pair. They are basically 'username' and 'password' to the service principle based authentification process. You need to generate them in Azure Portal, and give it 'Contributor' permissions to the resource group of your workspace. That ensures your Airflow connection can read/write your Azure ML resources to facilitate workloads. Please follow the 3 simple steps below to set them up.

#
To create a service principal, you need to follow 3 simple steps:
* Create a ```Client ID```. Follow instruction from the "Register an application with Azure AD and create a service principal" section of Azure guide [howto-create-service-principal-portal](https://learn.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal). ```Application ID```, aka ```Client ID```, is the unique identifier of this service principal.
* Create a ```Secret```. You can create a ```Secret``` under this application in the Azure Portal following the instructions in the "Option 2: Create a new application secret" section of [this instruction](https://learn.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal). Once a ```secret``` is successfully created, you will not be able to see the value. So we recommend you store your secret into Azure Key Vault, following [this instruction](https://learn.microsoft.com/en-us/azure/key-vault/secrets/quick-create-portal).
* Give this Service Principal ```Contribtor``` access to your Azure Machine Learning ```Resource Group```. Repeat the instruction form the item 7 above and land on your workspaces' resource page and click on the ```Resource Group```. From the left hand panel, select ```Access Control (IAM)``` and assign ```Contributor``` role to the the Application from above. This step is important. Without it, your Airflow will not have the necessary write access to necessary resources to create compute clusters, to execute training workloads, or to upload data, etc. Here is [an instruction to assign roles](https://learn.microsoft.com/en-us/azure/role-based-access-control/role-assignments-portal).

**Note**

If "Azure Machine Learning" is missing from the dropdown in step 3 above, it means ```airflow-providers-azure-machinelearning``` package is not successfully installed. You can follow instructions in the [Installation section](#Installation) to install it, and use commands like ``pip show airflow-provider-azure-machinelearning``` in the Airflow webserver container/machine to verify the package is installed correctly.

You can have many connections in one Airflow instance for different Azure Machine Learning workspaces. You can do this to:
1. Orchestrate workloads across multiple workspace/subscription from 1 single DAG.
2. Achieve isolation between different engineers' workload.
3. Achieve isolation between experimental and production environments.

The instructions above are for adding a connection via the Airflow UI. You can also do so via the Airflow Cli. You can find more examples of how to do this via Cli at [Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html). Below is an example Airflow command:
```bash
airflow connections add \
  --conn-type "azure_machine_learning" \
  --conn-description "[Description]" \
  --conn-host "schema" \
  --conn-login "[Client-ID]" \
  --conn-password "[Secret]" \
  --conn-extra '{"extra__azure_machine_learning__tenantId": "[Tenant-ID]", "extra__azure_machine_learning__subscriptionId": "[Subscription-ID]", "extra__azure_machine_learning__resource_group_name": "[Resource-Group-Name]", "extra__azure_machine_learning__workspace_name": "[Workspace-Name]"}' \
  "[Connection-ID]"
```
# Examples

Check out [example_dags](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/airflow_provider_azure_machinelearning/example_dags) on how to make use of this provider package. If you do not have a running Airflow instance, please refer to [example docker containers](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/dev/), or [Apache Airflow documentations\)https://airflow.apache.org/).

# Dev Environment

To build this package, run its tests, run its linting tools, etc, you will need following:
- Via pip: ```pip install -r dev/requirements.txt```
- Via conda: ```conda env create -f dev/environment.yml```

# Running the tests and linters
- All tests are in [tests](https://github.com/Azure/airflow-provider-azure-machinelearning/tree/main/tests/) folder. To run them, from this folder, run ```pytest```
- This repo uses [black](https://github.com/psf/black), [flake8](https://github.com/PyCQA/flake8), and [isort](https://github.com/PyCQA/isort) to keep coding format consistent. From this folder, run ```black .```, ```isort .```, and ```flake8```.

# Issues

Please submit issues and pull requests in our official repo: https://github.com/azure/airflow-provider-azure-machinelearning.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
