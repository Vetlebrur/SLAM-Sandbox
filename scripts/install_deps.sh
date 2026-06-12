#!/bin/bash
set -e

echo "==> Installing SLAM C++ dependencies via pacman..."
sudo pacman -S --needed \
    cmake \
    git \
    eigen3 \
    ceres-solver \
    google-glog \
    suitesparse

echo ""
echo "==> lz4 is already installed (used for bag decompression)."
echo "==> Done. Run scripts/build.sh to compile."
