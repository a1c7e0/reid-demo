#!/usr/bin/env bash
# Download the runtime ONNX models (face detection, face embedding, ReID).
set -euo pipefail

mkdir -p /models

fetch() {
  echo "[download] $2"
  curl -fL --retry 3 --retry-delay 2 -o "$2" "$1"
}

fetch "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx" \
      /models/yunet_2023mar.onnx
fetch "https://huggingface.co/Dao-Dung/auraface-v1-int8/resolve/main/glintr100_int8.onnx" \
      /models/glintr100_int8.onnx
fetch "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx" \
      /models/osnet_x0_25_msmt17.onnx
# default face embedding: MobileFaceNet WebFace600K (w600k_mbf) - wins the
# clustering objective (asymmetric: mixing heavy, splitting free) in
# scripts/eval/calib_bench.py; 14MB, same family as Ente's own model
fetch "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip" \
      /tmp/buffalo_sc.zip
cd /tmp && python -m zipfile -e buffalo_sc.zip /tmp/bsx && cp /tmp/bsx/w600k_mbf.onnx /models/ && rm -rf buffalo_sc.zip /tmp/bsx && cd /
fetch "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx" \
      /models/w600k_r50.onnx
# default face embedding (since lvface_t): LVFace-T Glint360K (MIT, ICCV 2025),
# 76.7 MB - wins the matched-purity comparison in docs/MODEL-ZOO.md (fewer
# look-alike mixes than w600k_mbf at the same purity floor) while staying
# inside the edge budget
fetch "https://huggingface.co/bytedance-research/LVFace/resolve/main/LVFace-T_Glint360K/LVFace-T_Glint360K.onnx" \
      /models/LVFace-T_Glint360K.onnx
echo "bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3  /models/LVFace-T_Glint360K.onnx" | sha256sum -c -

ls -la /models
