import os
import numpy as np
import jax
import jax.numpy as jnp
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
import matplotlib.pyplot as plt


class HeatConduction(Problem):
    def __init__(self, *args, b=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.b = b

    def custom_init(self):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def conductivity(gradT, theta_k):
            return theta_k * gradT
        return conductivity
    
    def get_mass_map(self):
        def mass_map(u, x, theta_k):
            return -jnp.array([self.b])
        return mass_map
    
    def get_surface_maps(self):
        def surface_map(u, x):
            return -jnp.array([0])
        return [surface_map, surface_map]
    
    def set_params(self, params):
        full_params = jnp.ones((self.fes[0].num_cells, params.shape[1]))
        full_params = full_params.at[self.fes[0].flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fes[0].num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


# --- k_eff ---
def k_eff_fn(rho, k0, k1, dy, T_top, T_bot, beta, fn):
    """
    计算有效热导率，并包含一个用于可视化温度场的内部函数。
    """
    rho_proj = (jnp.tanh(beta / 2) + jnp.tanh(beta * (rho - 0.5))) / (2 * jnp.tanh(beta / 2))
    k = k0 + rho_proj * (k1 - k0)
    # --- 您的原始代码开始 ---
    # p = 3.0
    # k = k0 + rho**p * (k1 - k0)
    
    # 注意：这里的 flatten顺序'F'必须和求解器内部节点排序的方式匹配！
    k_vec = k.flatten(order='F').reshape((-1, 1))
    
    sol = fn(k_vec)
    
    # 关键步骤：将求解器输出的一维向量重塑为二维温度场
    T = sol[0].reshape((Ny + 1, Nx + 1), order='F') # 尝试使用'F' order

    # 这里的切片是为了在每个单元格中心计算通量
    T_up = T[:-1, :-1]
    T_down = T[1:, :-1]
    
    q = k * (T_up - T_down) / dy
    q_avg = jnp.mean(q)
    k_eff = - q_avg * Ly / (T_top - T_bot) # keep the k positive

    return k_eff

if __name__ == '__main__':
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    Nx, Ny = 64, 64
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    try:
        foam_np = np.load('therm/2d_voxel.npy')
    except FileNotFoundError:
        print("no .npy file")
        x, y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
        foam_np = (np.sin(x * 10 * np.pi) * np.sin(y * 10 * np.pi) > 0).astype(int)

    foam = jnp.array(foam_np).astype(jnp.float32)

    # --- boundary conditions ---
    def left(point): return jnp.isclose(point[0], 0., atol=1e-5)
    def right(point): return jnp.isclose(point[0], Lx, atol=1e-5)
    def bottom(point): return jnp.isclose(point[1], 0., atol=1e-5)
    def top(point): return jnp.isclose(point[1], Ly, atol=1e-5)

    def dirichlet_val_top(point): return 100.
    def dirichlet_val_bottom(point): return 0.

    location_fns_dirichlet = [top, bottom]
    value_fns = [dirichlet_val_top, dirichlet_val_bottom]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    location_fns = [left, right]

    problem = HeatConduction(mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem)

    T_top = 100.
    T_bot = 0.
    rho_input = foam.astype(jnp.float32)
    k_eff_value = k_eff_fn(rho_input, k0=0.1, k1=100., T_top=100., T_bot=0., fn=fwd_pred)
    print(f"等效导热系数: {k_eff_value}")

    grad_fn = jax.grad(k_eff_fn)
    grad_val = grad_fn(rho_input, k0=0.1, k1=100., T_top=100., T_bot=0., fn=fwd_pred)

    print(f"梯度形状: {grad_val.shape}")
    print(f"梯度最大绝对值: {jnp.max(jnp.abs(grad_val))}")

    if jnp.all(jnp.isclose(grad_val, 0.)):
        print("\n警告：梯度仍然为零。")
    else:
        print("\n成功！梯度不再为零。")
        print("梯度预览 (前5x5):")
        print(grad_val[:5, :5])

    # Store the solution to local file.
    # save_dir = "./sol"
    # filename = "T_ver.vtu"
    # vtk_path = os.path.join(save_dir, filename)
    # save_sol(problem.fes[0], sol_list[0], vtk_path,
    #          cell_infos=[('k', k_vec[:,0])])


    # --- 新增：绘图函数 ---
    def plot_results(original_structure, gradient, k_eff):
        """
        将原始结构和梯度场并排绘制。
        
        Args:
            original_structure (np.ndarray): 原始的密度场 (0-1)。
            gradient (np.ndarray): 计算出的梯度场。
            k_eff (float): 计算出的等效导热系数。
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制原始结构
        # 使用 'gray_r' colormap，这样 k1 (值为1) 是黑色，k0 (值为0) 是白色
        im1 = axes[0].imshow(original_structure, cmap='gray_r', origin='lower')
        axes[0].set_title('Original Structure (rho)')
        axes[0].axis('off')

        # 绘制梯度场
        # 使用 'coolwarm' colormap，红色表示正梯度，蓝色表示负梯度
        im2 = axes[1].imshow(gradient, cmap='coolwarm', origin='lower')
        axes[1].set_title('Gradient of k_eff w.r.t. rho')
        axes[1].axis('off')
        
        # 为梯度图添加颜色条
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 添加总标题
        fig.suptitle(f'k_eff = {k_eff:.4f}', fontsize=16)
        
        # 调整布局并显示图像
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # --- 调用绘图函数 ---
    # 将 JAX 数组转换为 NumPy 数组以便 matplotlib 处理
    plot_results(np.array(rho_input), np.array(grad_val), k_eff_value)