import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional
from scipy.stats import entropy
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
import argparse
import os
import shutil


model = torch.hub.load(
                    'CarDS-Yale/PanEcho', 
                    'PanEcho', 
                    force_reload=True,
                    clip_len=32,
                    trust_repo=True  # Add trust_repo parameter
                )


dummy_input = torch.randn(1, 3, 32, 224, 224).to('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to('cuda').eval()

# Hook to capture feature extractor output
feature_output = None

def hook_fn(module, input, output):
    global feature_output
    feature_output = output

# Register hook on the encoder (before classification heads)
hook = model.encoder.register_forward_hook(hook_fn)

output = model(dummy_input)
print("Model loaded and output generated successfully.")

# Print the output structure to identify feature extractor output
if isinstance(output, dict):
    print("\nModel outputs:")
    for key, value in output.items():
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")
    
    # Now we have the feature extractor output captured by the hook
    if feature_output is not None:
        print(f"\nCaptured feature extractor output shape: {feature_output.shape}")
        print(f"Feature extractor output (first few values): {feature_output.flatten()[:10]}")
    
    # Remove the hook to clean up
    hook.remove()

# let's print the model architecture
# print("\nModel architecture:")
# for name, layer in model.named_modules():
#     # if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.Linear):
#     print(f"{name}: {layer}")