
# Environment Variables
export PROJECT='text-analysis-323506'
export REGION='asia-east1'
export BUCKET='text-analysis-323506'
export MACHINE_TYPE=n1-highmem-4
export ACCELERATOR_TYPE=NVIDIA_TESLA_K80
export ACCELERTOR_COUNT=1
export REPLICA_COUNT=1
export EXECUTE_IMAGE_URI='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest'
export LOCAL_PACKAGE_PATH='/home/jupyter/Distributed-Training-with-Tensorflow'
export PYTHON_MODULE='trainer.task'
export JOBNAME=distributed_training_$(date -u +%y%m%d_%H%M%S)

# gcloud configurations
gcloud config set project $PROJECT
gcloud config set compute/region $REGION

# Submit training job  
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=$JOBNAME \
    --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERTOR_COUNT,executor-image-uri=$EXECUTE_IMAGE_URI,local-package-path=$LOCAL_PACKAGE_PATH,python-module=$PYTHON_MODULE \
    --args=--data-dir=gs://text-analysis-323506/data,--save-dir=gs://text-analysis-323506/saved_model,--batch-size=1024

# Without accelerators
# gcloud ai custom-jobs create \
#     --region=$REGION \
#     --display-name=$JOBNAME \
#     --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,executor-image-uri=$EXECUTE_IMAGE_URI,local-package-path=$LOCAL_PACKAGE_PATH,python-module=$PYTHON_MODULE \
#     --args=--data-dir=gs://text-analysis-323506/data,--save-dir=gs://text-analysis-323506/saved_model,--batch-size=1024