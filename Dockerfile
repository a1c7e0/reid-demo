# syntax=docker/dockerfile:1

# ------------------------------------------------------------- js gate ---
# Fail the build on a JS syntax error. The UI is ES modules that the browser
# compiles at runtime, so a stray brace would otherwise only surface as a
# blank page in production (this happened: a long-press patch left one `}` too
# many and the whole app stopped mounting).
FROM node:26-alpine AS jscheck
WORKDIR /js
COPY app/static/js/ ./js/
RUN for f in js/*.js; do node --input-type=module --check < "$f" || exit 1; done

# ---------------------------------------------------------------- stage 1 ---
# Build-time only: download ONNX models and export the YOLO11n person
# detector (with NMS embedded) so the runtime image never needs PyTorch.
FROM python:3.12-slim AS models
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/download_models.sh scripts/export_detector.py scripts/export_reid.py \
     scripts/export_reid_zoo.py scripts/verify_models.py ./
RUN chmod +x download_models.sh && ./download_models.sh
# Model definition only (pure PyTorch); the torchreid package itself needs
# Cython/GCC to build, so export_reid.py loads osnet.py standalone.
RUN git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git /build/deep-person-reid

# CPU-only PyTorch keeps CUDA wheels out of the builder.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir ultralytics onnxruntime numpy pillow onnxscript && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless && \
    python -c "import cv2; print('cv2', cv2.__version__)"

# OSNet-AIN x1.0 (MSMT17) is the app's default body embedder, but only the
# torch checkpoint is published - export it here, verify the download by
# hash and let the parity assert in the script (torch vs onnx, <1e-4) run.
# (the checkpoint ships under torchreid's long run-name; verified against
#  the local export by sha256: 8a07e8da...919f)
RUN mkdir -p /build/zoo && \
    curl -fL --retry 3 --retry-delay 2 -o /build/zoo/osnet_ain_x1_0_msmt17.pth \
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth" && \
    echo "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f  /build/zoo/osnet_ain_x1_0_msmt17.pth" | sha256sum -c - && \
    ZOO=/build/zoo OSNET_AIN_PY=/build/deep-person-reid/torchreid/models/osnet_ain.py \
        python export_reid_zoo.py default && \
    cp /build/zoo/osnet_ain_x1_0_msmt17.onnx /models/

RUN python export_detector.py && python export_reid.py && python verify_models.py

# ---------------------------------------------------------------- stage 2 ---
FROM python:3.12-slim
ENV DATA_DIR=/data \
    MODEL_DIR=/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

COPY --from=models /models /models
# pull the JS syntax gate into this stage's build so a broken module fails
# the image build instead of rendering a blank page in production
RUN --mount=from=jscheck,target=/js-check true
COPY app ./app

RUN mkdir -p /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
