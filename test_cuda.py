import torch
from numba import njit, cuda

# Torch test
x = torch.tensor([1.0, 2.0, 3.0])
print("Torch tensor:", x)
print("Torch tensor sum:", x.sum().item())
print("Torch CUDA available:", torch.cuda.is_available())

# Numba CPU test
@njit
def add(a, b):
    return a + b

print("Numba add(2, 3):", add(2, 3))

# Numba CUDA test
if cuda.is_available():
    @cuda.jit
    def add_kernel(a, b, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i]

    import numpy as np
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    out = np.zeros_like(a)
    add_kernel[1, a.size](a, b, out)
    print("Numba CUDA add_kernel([1,2,3],[4,5,6]):", out)
else:
    print("Numba CUDA not available.")