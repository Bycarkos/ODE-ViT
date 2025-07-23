#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python main_lkis.py -c estimate_lkis.yaml -cp configs/koopman/
