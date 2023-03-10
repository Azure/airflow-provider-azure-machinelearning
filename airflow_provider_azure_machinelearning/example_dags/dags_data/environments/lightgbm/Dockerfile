FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20220930.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lightgbm
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.7 pip=20.2.4

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_TENSORFLOW=1 \
    pip install 'matplotlib>=3.3,<3.4' \
                'psutil>=5.8,<5.9' \
                'tqdm>=4.59,<4.60' \
                'pandas>=1.1,<1.2' \
                'numpy>=1.10,<1.20' \
                'scipy~=1.5.0' \
                'scikit-learn~=0.24.1' \
                'xgboost~=1.4.0' \
                'lightgbm~=3.2.0' \
                'dask~=2021.10.0' \
                'distributed~=2021.10.0' \
                'dask-ml~=2021.10.17' \
                'adlfs~=0.7.0' \
                'ipykernel~=6.0' \
                'debugpy~=1.6.3' \
                'azureml-core==1.47.0' \
                'azureml-defaults==1.47.0' \
                'azureml-mlflow==1.47.0' \
                'azureml-telemetry==1.47.0'

# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
