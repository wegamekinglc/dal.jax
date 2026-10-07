#!/usr/bin/env bash
# Build the pinned native reference into a dedicated directory and Python environment.
set -euo pipefail

oracle_root="${1:?Usage: build_dal_oracle.sh BUILD_ROOT PYTHON}"
oracle_python="${2:?Pass the Python executable of the target virtual environment}"
oracle_revision=4feabe89b105e0a3883fd4e2d74a2d70d618d8c7
mkdir -p "$oracle_root"
oracle_root="$(cd "$oracle_root" && pwd)"
oracle_source="$oracle_root/source"
oracle_prefix="$oracle_root/install"
if [[ ! -d "$oracle_source/.git" ]]; then
    git init "$oracle_source"
    git -C "$oracle_source" remote add origin https://github.com/wegamekinglc/Derivatives-Algorithms-Lib.git
    git -C "$oracle_source" fetch --depth 1 origin "$oracle_revision"
    git -C "$oracle_source" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$oracle_source" rev-parse HEAD)" != "$oracle_revision" ]]; then
    echo "Oracle checkout must be $oracle_revision; choose a fresh BUILD_ROOT." >&2
    exit 1
fi
git -c url.https://github.com/.insteadOf=git@github.com: -C "$oracle_source" submodule update --init --depth 1 \
    dal-cpp/externals/rapidjson dal-cpp/externals/machinist dal-cpp/externals/eigen
cmake -S "$oracle_source" -B "$oracle_root/build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$oracle_prefix" \
    -DBUILD_SHARED_LIBS=OFF -DDAL_BUILD_PUBLIC=ON -DDAL_BUILD_PYTHON=OFF \
    -DDAL_BUILD_EXCEL=OFF -DDAL_BUILD_EXCEL_PORTABLE_TESTS=OFF \
    -DDAL_CPP_BUILD_TESTS=OFF -DDAL_CPP_BUILD_EXAMPLES=OFF -DDAL_CPP_BUILD_BENCHMARKS=OFF \
    -DDAL_PUBLIC_BUILD_TESTS=OFF -DDAL_ENABLE_NATIVE_ARCH=OFF \
    -DDAL_USE_XAD_AAD=OFF -DDAL_USE_CODIPACK_AAD=OFF -DDAL_USE_ADEPT_AAD=OFF
cmake --build "$oracle_root/build" --target dal_public --parallel "${DAL_ORACLE_JOBS:-4}"
cmake --install "$oracle_root/build"
uv pip install --python "$oracle_python" "$oracle_source/dal-python" \
    --config-setting "cmake.define.DAL_INSTALL_PREFIX=$oracle_prefix" \
    --config-setting cmake.define.DAL_PYTHON_BUILD_TESTS=OFF
