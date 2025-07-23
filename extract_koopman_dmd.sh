#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python estimate_koopman_classification.py -c estimate_dmd_from_classification.yaml -cp configs/koopman/
