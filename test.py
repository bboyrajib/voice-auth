import torch
print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())
print('CUDA version    :', torch.version.cuda)
print('cuDNN version   :', torch.backends.cudnn.version())
if torch.cuda.is_available():
    print('GPU             :', torch.cuda.get_device_name(0))
    print('VRAM            :', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
else:
    print('--- Checking why CUDA is unavailable ---')
    import subprocess
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print('nvidia-smi:', r.stdout or r.stderr)