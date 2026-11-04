#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m pairception.train --seed 42
CUDA_VISIBLE_DEVICES=0 python -m pairception.evaluate --model pairception --seed 42
