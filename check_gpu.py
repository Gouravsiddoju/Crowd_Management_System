import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor move
        x = torch.tensor([1.0]).cuda()
        print(f"Tensor on GPU: {x.is_cuda}")
    else:
        print("CUDA IS NOT AVAILABLE from PyTorch.")
except Exception as e:
    print(f"Error checking CUDA: {e}")
