#!/usr/bin/env sh
pyenv local retino

python run_preprocessing.py \
  --denoise-str nordic-mat_11_5 \
  --build-code rd dr d D Dr rD\
  --sub 6 \
