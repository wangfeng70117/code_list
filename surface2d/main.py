import taichi as ti
import numpy as np
from surface2d.const import const
from surface2d.mpm_solver import MPMSolver

ti.init(arch=ti.gpu)

n_grid = 128
draw_surface = 1
mpm_solver = MPMSolver(30000, n_grid, 1)
mpm_solver.add_cube(ti.Vector([0.4, 0.6]), 0.12, const.MATERIAL_FLUID, 6000)

gui = ti.GUI("Tension Code", res=512, background_color=0x112F41)
colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
colors2 = np.array([0x00ff00, 0xff00ff], dtype=np.uint32)
run = 1
while gui.running:
    if run == 1:
        mpm_solver.run()

    particle_num = mpm_solver.base_particle_num[None]
    particles = mpm_solver.particles
    gui.circles(particles.position.to_numpy()[:particle_num], radius=1.5,
                color=colors[particles.material.to_numpy()[:particle_num]])

    if draw_surface:
        begin = mpm_solver.node_mc.edge.begin_point.to_numpy()
        end = mpm_solver.node_mc.edge.end_point.to_numpy()
        edge_num = mpm_solver.node_mc.edge_num[None]
        gui.lines(begin[: edge_num], end[:edge_num], radius=1)
    # gui.show(f'{gui.frame:06d}.png')
    gui.show()