# Visitor Counting Console - browser console on an NVIDIA GPU (Ubuntu host with the
# NVIDIA Container Toolkit installed). Build once, run with GPU access:
#   docker compose up --build            # http://localhost:8780, video mode
#   MODE=live docker compose up          # RTSP cameras from config/sources.json
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf YOLO_CONFIG_DIR=/models/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch/torchvision come from the base image (CUDA 12.8); install the rest
RUN grep -viE '^(torch|torchvision)\b' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY humanmonitor/ humanmonitor/
COPY web/ web/
COPY config/ config/
COPY app.py webapp.py download_demo_videos.py export_tensorrt.py ./

EXPOSE 8780
VOLUME ["/models", "/app/output", "/app/sample_data"]
ENTRYPOINT ["python", "webapp.py", "--host", "0.0.0.0", "--port", "8780"]
CMD ["--mode", "video", "--model", "/models/yolo11s.pt"]
