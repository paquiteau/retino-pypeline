#!/usr/bin/env sh
pyenv local retino

python run_preprocessing.py --denoise-config optimal-fro_11_5 --build-code v r rd dr d --sub 6
