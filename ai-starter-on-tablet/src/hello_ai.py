import sys
print("Hello AI from your Galaxy Tab workflow!")
print("Python:", sys.version)
try:
    import torch, transformers
    print("Torch:", torch.__version__)
    print("Transformers:", transformers.__version__)
except Exception as e:
    print("Optional libs not installed yet:", e)
