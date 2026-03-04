#!/bin/bash
# Script to download CASIA 2.0 dataset for Image Manipulation Detection
# Requirement: Kaggle API installed and configured (pip install kaggle)

set -e

DATASET_ROOT="/home/uslib/quynhhuong/datasets/CASIA2"
mkdir -p "$DATASET_ROOT"

echo "Attempting to download CASIA 2.0 from Kaggle..."
kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset -p "$DATASET_ROOT"

echo "Unzipping dataset..."
unzip -q "$DATASET_ROOT/casia-20-image-tampering-detection-dataset.zip" -d "$DATASET_ROOT"
rm "$DATASET_ROOT/casia-20-image-tampering-detection-dataset.zip"

echo "Downloading Corrected Ground Truth (MANDATORY for CASIA 2.0)..."
cd "$DATASET_ROOT"
git clone https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth.git corrected_gt

echo "Merging corrected ground truth..."
# The dataset has Au and Tp folders. Corrected GT has Tp masks.
# Corrected GT structure: corrected_gt/GT/
mkdir -p "$DATASET_ROOT/GT"
cp -r corrected_gt/GT/* "$DATASET_ROOT/GT/"

echo "Cleanup..."
rm -rf corrected_gt

echo "CASIA 2.0 dataset initialized at $DATASET_ROOT"
ls -d "$DATASET_ROOT"/*/
