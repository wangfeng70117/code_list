import taichi as ti


@ti.data_oriented
class LevelSet:
    def __init__(self, diff_n_grid):
        self.diff_n_grid = diff_n_grid
        self.diff_dx = 1 / self.diff_n_grid
        self.diff_inv_dx = 1 / self.diff_dx
        self.radius = self.diff_dx
        self.sign_distance_field = ti.field(ti.f32, shape=(self.diff_n_grid, self.diff_n_grid))

    # 生成level set隐式曲面
    @ti.kernel
    def gen_level_set(self, particle_position: ti.template(),
                      create_particle_num: int):
        for i, j in ti.ndrange(self.diff_n_grid, self.diff_n_grid):
            min_dis = 10.0
            node_pos = ti.Vector([i * self.diff_dx, j * self.diff_dx])
            for I in range(create_particle_num):
                distance = (particle_position[I] - node_pos).norm() - self.radius
                if distance < min_dis:
                    min_dis = distance
            self.sign_distance_field[i, j] = min_dis
