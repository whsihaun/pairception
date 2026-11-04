#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m pairception.evaluate --model aerialvg
CUDA_VISIBLE_DEVICES=0 python -m pairception.evaluate --model frozen_detector
CUDA_VISIBLE_DEVICES=0 python -m pairception.evaluate --model pairception
