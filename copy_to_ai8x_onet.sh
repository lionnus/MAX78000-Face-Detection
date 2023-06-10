#!/bin/bash

# Copy widerfaceonet.yaml to /maxim7800-face-detection/ai8x/Exercise8/ai8x-synthesis/networks
cp ./ai8x_onet/widerfaceonet.yaml ../Exercise8/ai8x-synthesis/networks/

# Copy widerfacernet.yaml to /maxim7800-face-detection/ai8x/Exercise8/ai8x-synthesis/networks
cp ./ai8x_onet/widerfacernet.yaml ../Exercise8/ai8x-synthesis/networks/

# Copy widerfaceonet.py to /Exercise8/ai8x-training/model
cp ./ai8x_onet/widerfacenet.py ../Exercise8/ai8x-training/models/

# Copy wider_faces.py to /Exercise8/ai8x-training/dataset
cp ./ai8x_onet/widerfaces.py ../Exercise8/ai8x-training/datasets/

# Copy qat_policy_widerfacenet.yaml to /Exercise8/ai8x-training/policies
cp ./ai8x_onet/qat_policy_widerfacenet.yaml ../Exercise8/ai8x-training/policies/
