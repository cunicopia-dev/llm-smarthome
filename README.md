# llm-smarthome
 Cunicopian SmartHome Protocol (CSHP)

sudo apt install cmake
sudo apt install onnx
sudo apt install protobuf-compiler


sudo docker run --rm --gpus all --name ollama -d -p 11434:11434 ollama/ollama
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main