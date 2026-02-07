#!/bin/bash
# Test stacking with CPU-only mode (disables OpenCL GPU acceleration)
# Use this if you encounter OpenCL errors during stacking

export IMAGESTACKER_STACKING_CPU=1

./target/release/imagestacker --import testimages-focus-stack-pyramid
