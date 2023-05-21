#!/bin/bash

# Copy widerfacenet.yaml to /maxim7800-face-detection/ai8x/Exercise8/ai8x-synthesis/networks
cp ./ai8x/widerfacenet.yaml ../Exercise8/ai8x-synthesis/networks/

# Copy wider_face_net.py to /Exercise8/ai8x-training/model
cp ./ai8x/widerfacenet.py ../Exercise8/ai8x-training/models/

# Copy wider_faces.py to /Exercise8/ai8x-training/dataset
cp ./ai8x/widerfaces.py ../Exercise8/ai8x-training/datasets/

# Copy qat_policy_widerfacenet.yaml to /Exercise8/ai8x-training/policies
cp ./ai8x/qat_policy_widerfacenet.yaml ../Exercise8/ai8x-training/policies/