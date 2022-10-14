#!/usr/bin/env bash

# Create a Singularity image with everything needed (SPM12, FSL5, etc)
#
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR=$(realpath "$SCRIPT_DIR/../build")

[ -d "$BUILD_DIR" ] || mkdir -p "$BUILD_DIR"

pushd "$BUILD_DIR" || exit

[ -f "$BUILD_DIR/neurodocker.sif" ] || singularity build neurodocker.sif docker://repronim/neurodocker:0.7.0

singularity run $BUILD_DIR/neurodocker.sif generate singularity \
    -p apt -b debian:stretch-slim \
    --install git \
    --fsl version=5.0.11 \
    --spm version=r7771 \
    --miniconda \
        version=latest \
        create_env=False \
        pip_install="git+https://github.com/paquiteau/patch-denoising.git \
                     git+https://github.com/paquiteau/retino-pypeline.git" \
> retino-pypeline.rec

[ -d "$BUILD_DIR/tmp" ] || mkdir "$BUILD_DIR/tmp"
[ -d "$BUILD_DIR/cache" ] || mkdir "$BUILD_DIR/cache"

sudo SINGULARITY_TMPDIR="$BUILD_DIR/tmp" singularity build retino.sif retino-pypeline.rec

popd || return
