import numpy as np
import taichi as ti
import time
import math

ti.init(arch=ti.cpu)

@ti.data_oriented
class MGPCG:
    '''
    Grid-based MGPCG solver for the 2D poisson equation: \nabla u = f
    (right-hand-side f can be set in self.init)
    We use matrix-free to avoid the storage of matrix A
    This solver only runs on CPU and CUDA backends since it
    requires the 'pointer' SNode
    '''
    def __init__(self, N=128, n_mg_levels=4):
        '''
        N: Grid resolution
        n_mg_levels: Number of multigrid levels
        '''

        # use Multigrid methods as preconditioner of conjugate gradient
        self.use_multigrid = True

        # Dimesionality of the fields
        self.dim = 2

        self.N = N
        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.h = 1.0    # grid pace

        self.N_ext = self.N // 2  # number of ext cells so that the total grid size is still power of 2
        self.N_tot = 2 * self.N

        # setup sparse simulation data arrays
        self.r = [ti.field(dtype=ti.f32)
                for _ in range(self.n_mg_levels)] # r
        self.res = [ti.field(dtype=ti.f32)
                for _ in range(self.n_mg_levels)] # r - Az
        self.z = [ti.field(dtype=ti.f32)
                for _ in range(self.n_mg_levels)] # M^-1 self.r
        self.x = ti.field(dtype=ti.f32) # solution
        self.p = ti.field(dtype=ti.f32) # conjugate gradient
        self.Ap = ti.field(dtype=ti.f32) # matrix-vector product
        self.alpha = ti.field(dtype=ti.f32)
        self.beta = ti.field(dtype=ti.f32)
        self.sum = ti.field(dtype=ti.f32)

        indices = ti.ij
        self.grid = ti.root.pointer(indices, [self.N_tot // 4]).dense(
                indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices, [self.N_tot // (4 * 2**l)]).dense(
                    indices, 4).place(self.r[l], self.res[l], self.z[l])

        ti.root.place(self.alpha, self.beta, self.sum)

    @ti.func
    def init_r(self, I, r_I): # activate sparse arrays and initialize
        I = I + self.N_ext
        self.r[0][I] = r_I
        self.res[0][I] = 0
        self.z[0][I] = 0
        self.x[I] = 0
        self.p[I] = 0
        self.Ap[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()): # initialize right-hand-side
        '''
        r: (ti.field) Unscaled right-hand-side
        k: (scalar) A scaling factor of the right-hand-side
        '''
        for I in ti.grouped(ti.ndrange(self.N, self.N)):
            self.init_r(I, k * r[I])

    @ti.func
    def get_x(self, I):
        I = I + self.N_ext
        return self.x[I]

    @ti.kernel
    def get_result(self) -> ti.f32:
        # Get the sum of elements in solution field.
        ret = 0.0
        for I in ti.grouped(ti.ndrange(self.N, self.N)):
            ret += self.get_x(I)
        return ret

    @ti.kernel
    def compute_Ap(self):
        for i, j in self.Ap:
            self.Ap[i, j] = (self.p[i, j-1] + self.p[i, j+1] + self.p[i-1, j] + self.p[i+1, j] -
                    4.0 * self.p[i, j]) / self.h**2

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.x):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.r[0]):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def compute_residual(self, l: ti.template()): # res[l] = r[l] - A z[l]
        for i, j in self.r[l]:
            self.res[l][i, j] = self.r[l][i, j] - (self.z[l][i, j-1] + self.z[l][i, j+1] +
                    self.z[l][i-1, j] + self.z[l][i+1, j] - 4.0 * self.z[l][i, j]) / self.h**2

    @ti.kernel
    def restrict(self, l: ti.template()):
        a = self.N_ext // 2**(l + 1)
        b = (self.N_ext + self.N) // 2**(l + 1)
        for i, j in ti.ndrange((a, b), (a, b)):
            self.r[l+1][i, j] = (self.res[l][2*i-1, 2*j-1] + self.res[l][2*i-1, 2*j+1] +
                    self.res[l][2*i+1, 2*j-1] + self.res[l][2*i+1, 2*j+1] + 2.0 * (
                        self.res[l][2*i, 2*j-1] + self.res[l][2*i, 2*j+1] + self.res[l][2*i-1, 2*j] +
                        self.res[l][2*i+1, 2*j]) + 4.0 * self.res[l][2*i, 2*j]) / 16

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for i, j in self.z[l]:
            self.z[l-1][2*i, 2*j] += self.z[l][i, j]
            self.z[l-1][2*i+1, 2*j] += (self.z[l][i, j] + self.z[l][i+1, j]) / 2
            self.z[l-1][2*i, 2*j+1] += (self.z[l][i, j] + self.z[l][i, j+1]) / 2
            self.z[l-1][2*i+1, 2*j+1] += (self.z[l][i, j] + self.z[l][i+1, j] +
                    self.z[l][i, j+1] + self.z[l][i+1, j+1]) / 4

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # red/black Gauss-Seidel smooth
        for i, j in self.r[l]:
            if (i + j) & 1 == phase:
                self.z[l][i, j] = (self.z[l][i, j-1] + self.z[l][i, j+1] + self.z[l][i-1, j] +
                        self.z[l][i+1, j] - self.h**2 * self.r[l][i, j]) / 4.0

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.compute_residual(l)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l + 1)
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self, max_iters=-1, eps=1e-12, abs_tol=1e-12, rel_tol=1e-12, verbose=False):
        '''
        max_iters: Specify the maximum iterations. -1 for no limit.
        eps: Specify a non-zero value to prevent ZeroDivisionError.
        abs_tol: Specify the absolute tolerance of loss
        rel_tol: Specify the tolerance of loss relative to initial loss
        '''

        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]
        tol = max(abs_tol, initial_rTr * rel_tol)

        # self.r = b - Ax = b  since self.x = 0
        # self.p = self.r + 0 self.p = self.r
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # Conjugate Gradients
        iter = 0
        while max_iters == -1 or iter < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            if verbose:
                print(f'iter {iter}, ||residual||_2 = {math.sqrt(rTr)}')
            if rTr < tol:
                break

            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            iter += 1

        print(f"result: {self.get_result()}")

class MGPCG_Example(MGPCG):
    def __init__(self):
        super().__init__(N=128, n_mg_levels=3)

    @ti.kernel
    def init(self):       # initialize right-hand-side: f = 5cos(5 \pi x)cos(5 \pi y)
        for I in ti.grouped(ti.ndrange(self.N, self.N)):
            r_I = 5.0
            r_I *= ti.cos(5 * np.pi * I[0] / self.N)
            r_I *= ti.cos(5 * np.pi * I[1] / self.N)
            self.init_r(I, r_I)

    def run(self, verbose=False):
        self.init()
        self.solve(max_iters=400, verbose=verbose)

if __name__ == '__main__':
    solver = MGPCG_Example()
    t = time.time()
    solver.run(verbose=True)
    print(f'Solver time: {time.time() - t: .3f} s')
