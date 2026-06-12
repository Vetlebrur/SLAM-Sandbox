#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

mkdir -p "$BUILD"
cd "$BUILD"

cmake "$ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BUILD/bin" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

make -j"$(nproc)"

echo ""
echo "==> Build complete. Binaries in $BUILD/bin/"
echo ""
echo "Run tests:"
echo "  $BUILD/bin/test_eigen"
echo "  $BUILD/bin/test_ceres"
echo "  $BUILD/bin/test_sophus"
echo "  $BUILD/bin/test_bag  [path/to/file.bag]"
