import torch
import gc

# 1. Clear Python's garbage collector reference counters
gc.collect()

# 2. Free all unused cached memory blocks from PyTorch allocator
torch.cuda.empty_cache()
