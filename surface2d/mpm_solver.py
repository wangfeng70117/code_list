import taichi as ti
from surface2d.const import const
from surface2d.marching_squares import MarchingSquares
from surface2d.level_set import LevelSet


@ti.data_oriented
class MPMSolver:
    def __init__(self,
                 max_particle_num,
                 grid_num,
                 quality,
                 ):
        self.quality = quality  # Use a larger value for higher-res simulations
        self.max_particle_num = max_particle_num
        self.grid_num = grid_num * quality
        self.dx = 1 / self.grid_num
        self.inv_dx = float(self.grid_num)
        self.dt = 1e-4 / quality
        self.p_vol = (self.dx * 0.5) ** 2
        self.E, self.nu = 0.1e4, 0.14  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = self.E / (2 * (1 + self.nu)), self.E * self.nu / (
                (1 + self.nu) * (1 - 2 * self.nu))  # Lame parameters
        self.particles = ti.Struct.field({
            "position": ti.types.vector(2, ti.f32),
            "velocity": ti.types.vector(2, ti.f32),
            "F": ti.types.matrix(2, 2, ti.f32),
            "C": ti.types.matrix(2, 2, ti.f32),
            "mass": ti.f32,
            "material": ti.i32,
            "Jp": ti.f32
        }, shape=max_particle_num)
        self.node = ti.Struct.field({
            "node_m": ti.f32,
            "node_v": ti.types.vector(2, ti.f32),
        }, shape=(grid_num, grid_num))
        self.base_particle_num = ti.field(ti.i32, ())
        self.node_mc = MarchingSquares(self.grid_num)
        self.node_sdf = LevelSet(self.grid_num)

    @ti.kernel
    def reset_node(self):
        for I in ti.grouped(self.node):
            self.node[I].node_m = .0
            self.node[I].node_v = [.0, .0]

    @ti.kernel
    def P2G(self):
        for p in range(self.base_particle_num[None]):  # Particle state update and scatter to grid (P2G)
            base = (self.particles[p].position * self.inv_dx - 0.5).cast(int)
            fx = self.particles[p].position * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.particles[p].F = (ti.Matrix.identity(float, 2) + self.dt * self.particles[p].C) @ self.particles[
                p].F  # deformation gradient update
            h = ti.exp(10 * (1.0 - self.particles[p].Jp))  # Hardening coefficient: snow gets harder when compressed
            if self.particles[p].material == const.MATERIAL_SOLID:  # jelly, make it softer
                h = 5
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.particles[p].material == const.MATERIAL_FLUID:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.particles[p].F)
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                self.particles[p].Jp *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.particles[
                p].material == const.MATERIAL_FLUID:  # Reset deformation gradient to avoid numerical instability
                self.particles[p].F = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif self.particles[p].material == 2:
                self.particles[
                    p].F = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (self.particles[p].F - U @ V.transpose()) @ self.particles[
                p].F.transpose() + ti.Matrix.identity(float,
                                                      2) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.particles[p].mass * self.particles[p].C
            for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.node[base + offset].node_v += weight * (
                        self.particles[p].mass * self.particles[p].velocity + affine @ dpos)
                self.node[base + offset].node_m += weight * self.particles[p].mass

    @ti.kernel
    def grid_operator(self):
        for I in ti.grouped(self.node):
            i, j = I
            if self.node[i, j].node_m > 0:  # No need for epsilon here
                self.node[i, j].node_v /= self.node[i, j].node_m  # Momentum to velocity
                self.node[i, j].node_v[1] -= self.dt * 10  # gravity
                if i < 3 and self.node[i, j].node_v[0] < 0:
                    self.node[i, j].node_v[0] = 0
                if i > self.grid_num - 3 and self.node[i, j].node_v[0] > 0:
                    self.node[i, j].node_v[0] = 0
                if j < 3 and self.node[i, j].node_v[1] < 0:
                    self.node[i, j].node_v[1] = 0
                if j > self.grid_num - 3 and self.node[i, j].node_v[1] > 0:
                    self.node[i, j].node_v[1] = 0

    @ti.kernel
    def fluid_G2P(self):
        for p in range(self.base_particle_num[None]):  # grid to particle (G2P)
            if self.particles[p].material == const.MATERIAL_FLUID:
                base = (self.particles[p].position * self.inv_dx - 0.5).cast(int)
                fx = self.particles[p].position * self.inv_dx - base.cast(float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(float, 2)
                new_C = ti.Matrix.zero(float, 2, 2)
                for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
                    dpos = ti.Vector([i, j]).cast(float) - fx
                    g_v = self.node[base + ti.Vector([i, j])].node_v
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                self.particles[p].velocity, self.particles[p].C = new_v, new_C

    @ti.kernel
    def advance(self):
        for p in range(self.base_particle_num[None]):  # grid to particle (G2P)
            self.particles[p].position += self.dt * self.particles[p].velocity  # advection

    # particle type  为0：流体粒子  为1 边界粒子
    # material 为0：水 为1：固体
    @ti.kernel
    def add_cube(self, position: ti.template(), length: float, material: int, particle_num: int):
        print("Add Cube...")
        rho = 1
        if material == const.MATERIAL_SOLID:
            rho = 7

        for i in range(self.base_particle_num[None], self.base_particle_num[None] + particle_num):
            n = ti.atomic_add(self.base_particle_num[None], 1)
            self.particles[n].position = [ti.random() * length + position[0], ti.random() * length + position[1]]
            self.particles[n].material = material
            self.particles[n].velocity = ti.Matrix([0, 0])
            self.particles[n].F = ti.Matrix([[1, 0], [0, 1]])
            self.particles[n].mass = self.p_vol * rho
            self.particles[n].Jp = 1

    def run(self):
        for i in range(30):
            self.reset_node()
            # 必须先标记，然后计算边界
            self.P2G()
            # # 使用质量场构建mc
            # self.node_mc.update_field(self.node.node_m)
            # self.node_mc.create_mc()
            # 使用sdf构建mc
            self.node_sdf.gen_level_set(self.particles.position, self.base_particle_num[None])
            self.node_mc.update_field(self.node_sdf.sign_distance_field)
            self.node_mc.create_mc()
            self.grid_operator()
            self.fluid_G2P()
            self.advance()
