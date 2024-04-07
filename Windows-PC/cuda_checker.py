'''
To run YOLOv8 on a GPU, troubleshoot using this python program.

Check first using !nvidia-smi in terminal. Then, download the CUDA- and 
    CUDNN compatible software to your GPU.

Include the files from CUDNN zip file inside the CUDA folder.

Set these paths as path environment variables:
C:/Program Files/NVidia GPU Computing Toolkit/CUDA/v12.3/bin
C:/Program Files/NVidia GPU Computing Toolkit/CUDA/v12.3/lib/x64
C:/Program Files/NVidia GPU Computing Toolkit/CUDA/v12.3/include

Restart computer afterwards.

Usage:
python cuda_checker.py
'''

import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Check for CUDA device and set it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')