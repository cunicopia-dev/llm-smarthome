#!/bin/bash

# Define variables
VOLUME_NAME="ollama_data"
CONTAINER_NAME="ollama"
IMAGE_NAME="ollama/ollama"
PORT_MAPPING="11434:11434"
GPU_OPTION="--gpus all"
MODEL_DIR="/opt/ollama/models"  # Update this path based on where Ollama saves its models

# Create the Docker volume if it doesn't already exist
if [[ $(sudo docker volume ls -q --filter name=^${VOLUME_NAME}$) != ${VOLUME_NAME} ]]; then
    echo "Creating Docker volume named ${VOLUME_NAME}..."
    sudo docker volume create ${VOLUME_NAME}
else
    echo "Volume ${VOLUME_NAME} already exists."
fi

# Ensure container is not running before starting a new one
if [[ $(sudo docker ps -aq --filter name=^/${CONTAINER_NAME}$) ]]; then
    echo "Container ${CONTAINER_NAME} is found. Stopping and removing..."
    sudo docker stop ${CONTAINER_NAME}
    sudo docker rm ${CONTAINER_NAME}
fi

# Run the Docker container with the necessary options
echo "Starting Docker container ${CONTAINER_NAME}..."
sudo docker run --rm ${GPU_OPTION} --name ${CONTAINER_NAME} -d -p ${PORT_MAPPING} -v ${VOLUME_NAME}:${MODEL_DIR} ${IMAGE_NAME}

echo "Container ${CONTAINER_NAME} has been started successfully with volume ${VOLUME_NAME} mounted at ${MODEL_DIR}."


# Hello 