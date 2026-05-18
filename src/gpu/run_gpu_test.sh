#!/usr/bin/env bash
# run_gpu_tests.sh
# Build ring_buffer.so, cupti_layer.so, test_gpu, then run the G1 test.
# Run from project root: bash src/gpu/run_gpu_tests.sh

set -e

# Detect CUDA layout automatically
if [ -f "/usr/include/cupti.h" ]; then
    CUDA_INC="/usr/include"
    CUPTI_INC="/usr/include"
    CUDA_LIB="/usr/lib/x86_64-linux-gnu"
    CUPTI_LIB="/usr/lib/x86_64-linux-gnu"
elif [ -f "/usr/local/cuda/extras/CUPTI/include/cupti.h" ]; then
    CUDA_INC="/usr/local/cuda/include"
    CUPTI_INC="/usr/local/cuda/extras/CUPTI/include"
    CUDA_LIB="/usr/local/cuda/lib64"
    CUPTI_LIB="/usr/local/cuda/extras/CUPTI/lib64"
else
    echo "ERROR: cupti.h not found. Run: sudo apt install libcupti-dev"
    exit 1
fi

echo "=== Build: ring_buffer.so ==="
python3 src/cpp/build.py

echo ""
echo "=== Build: cupti_layer.so ==="
python3 src/gpu/build_gpu.py

echo ""
echo "=== Build: test_gpu ==="
nvcc -O2 -std=c++17 \
    -I"$CUDA_INC" \
    -I"$CUPTI_INC" \
    tests/test_gpu.cu \
    src/cpp/ring_buffer.so \
    src/gpu/cupti_layer.so \
    -L"$CUDA_LIB" \
    -L"$CUPTI_LIB" \
    -lcuda -lcupti \
    -Xlinker -rpath -Xlinker "$(pwd)/src/cpp" \
    -Xlinker -rpath -Xlinker "$(pwd)/src/gpu" \
    -Xlinker -rpath -Xlinker "$CUPTI_LIB" \
    -Xlinker -rpath -Xlinker "$CUDA_LIB" \
    -o tests/test_gpu

echo ""
echo "=== Run: test_gpu ==="
LD_LIBRARY_PATH="$(pwd)/src/cpp:$(pwd)/src/gpu:$CUPTI_LIB:$CUDA_LIB:$LD_LIBRARY_PATH" \
    ./tests/test_gpu