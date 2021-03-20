# MGPCG_Taichi

This code is based on [Taichi](https://taichi.readthedocs.io/en/stable/install.html)

Grid-based MGPCG solver for the 2D poisson equation: $\nabla u = f$
(right-hand-side f can be set in self.init)
We use matrix-free to avoid the storage of matrix A.
 
This solver only runs on CPU and CUDA backends since it requires the 'pointer' SNode.
