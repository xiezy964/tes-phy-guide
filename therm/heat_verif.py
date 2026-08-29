import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

# Matplotlib 用于生成图表
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. 物理和求解器参数设置
# --------------------------------------------------------------------------
K_STRUCTURE = 10.0      # 结构材料的导热系数
K_AIR = 0.1             # 空气（或基底）的导热系数
T_HOT = 100.0           # 上边界温度
T_COLD = 0.0            # 下边界温度

# 求解器参数
MAX_ITERATIONS = 2000
CONVERGENCE_TOL = 1e-6  # 已不再用于提前收敛，但保留参数
FD_EPS = 1e-3           # 有限差分扰动幅度（对密度/体素变量）

# --------------------------------------------------------------------------
# 2. 核心计算函数（取消提前收敛，固定迭代次数）
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['p'])
def calculate_keff_and_temp(voxel_structure, p=3.0):
    """
    计算给定结构的等效导热系数 k_eff 和最终温度场 T_final。
    取消提前收敛，固定执行 MAX_ITERATIONS 次迭代。

    Args:
        voxel_structure (jnp.array): 代表结构的二维数组 (0 for air, 1 for structure)。
        p (float): SIMP惩罚因子。

    Returns:
        tuple: (k_eff, T_final)
            - k_eff (float): 计算出的等效导热系数。
            - T_final (jnp.array): 最终的稳态温度场。
    """
    height, width = voxel_structure.shape
    
    # --- 步骤 A: 计算导热系数场 k_map ---
    # SIMP: k = k_min + rho^p * (k_max - k_min)
    k_map = K_AIR + (voxel_structure**p) * (K_STRUCTURE - K_AIR)

    # --- 步骤 B: 使用 fori_loop 固定迭代求解温度场 T ---
    T_init = jnp.full_like(k_map, (T_HOT + T_COLD) / 2.0)
    T_init = T_init.at[0, :].set(T_HOT)
    T_init = T_init.at[-1, :].set(T_COLD)

    def body_fun(i, T):
        # 各向同性导热的调和平均到面
        k_y_fwd = 2 * k_map * jnp.roll(k_map, -1, axis=0) / (k_map + jnp.roll(k_map, -1, axis=0))
        k_y_bwd = jnp.roll(k_y_fwd, 1, axis=0)
        k_x_fwd = 2 * k_map * jnp.roll(k_map, -1, axis=1) / (k_map + jnp.roll(k_map, -1, axis=1))
        k_x_bwd = jnp.roll(k_x_fwd, 1, axis=1)
        
        numerator = (k_y_bwd * jnp.roll(T, 1, axis=0) +
                     k_y_fwd * jnp.roll(T, -1, axis=0) +
                     k_x_bwd * jnp.roll(T, 1, axis=1) +
                     k_x_fwd * jnp.roll(T, -1, axis=1))
        denominator = k_y_bwd + k_y_fwd + k_x_bwd + k_x_fwd
        T_new = numerator / denominator
        
        # 施加边界条件（上热下冷，左右为周期/自然由 roll 模拟）
        T_new = T_new.at[0, :].set(T_HOT)
        T_new = T_new.at[-1, :].set(T_COLD)
        return T_new

    T_final = jax.lax.fori_loop(0, MAX_ITERATIONS, body_fun, T_init)
    
    # --- 步骤 C: 计算总热流 Q_total ---
    # 顶部界面热通量（单位长度，dx=dy=1）
    k_interface_top = 2 * k_map[0, :] * k_map[1, :] / (k_map[0, :] + k_map[1, :])
    q_y = k_interface_top * (T_final[0, :] - T_final[1, :])
    Q_total = jnp.sum(q_y)

    # --- 步骤 D: 计算等效导热系数 k_eff ---
    delta_T = T_HOT - T_COLD
    L = height  # 有效长度（y方向单元数），网格间距假设为1
    A = width   # 横截面积（x方向宽度），网格间距假设为1
    k_eff = (Q_total * L) / (A * delta_T)
    
    return k_eff, T_final

# --------------------------------------------------------------------------
# 3. 有限差分梯度（中心差分）计算
# --------------------------------------------------------------------------

def finite_difference_gradient(voxel_structure_np, p=3.0, eps=FD_EPS, batch=128):
    """
    使用有限差分（中心差分为主，边界退化为前/后向差分）计算对 voxel_structure 的梯度。
    为提高效率，按批次对像素进行计算，利用 vmap + jit。

    Args:
        voxel_structure_np (np.ndarray): float32/float64 2D 数组，取值通常在 [0,1]
        p (float): SIMP 惩罚因子
        eps (float): 扰动幅度
        batch (int): 批大小，用于向量化加速

    Returns:
        np.ndarray: 与输入同形状的数值梯度数组
    """
    voxel0 = jnp.array(voxel_structure_np, dtype=jnp.float32)
    H, W = voxel0.shape
    N = H * W

    # 基函数：给定 perturbed voxel，返回 k_eff
    @partial(jax.jit, static_argnames=['p'])
    def k_eff_only(vox, p=3.0):
        keff, _ = calculate_keff_and_temp(vox, p=p)
        return keff

    # 生成单位基方向的扰动掩码
    indices = jnp.arange(N)

    def one_fd(idx, vox):
        # 将一维 idx 映射到 (i, j)
        i = idx // W
        j = idx % W

        # 取中心差分，注意边界退化
        def clamp01(x):
            return jnp.clip(x, 0.0, 1.0)

        v_ij = vox[i, j]
        # 尝试中心差分
        vox_plus  = vox.at[i, j].set(clamp01(v_ij + eps))
        vox_minus = vox.at[i, j].set(clamp01(v_ij - eps))

        # 如果中心差分两侧都合法（这里我们用了 clamp，始终合法），直接中心差分
        # 为保持定义严格：若 clamp 生效导致两者差分间距 < 2*eps，则退化为单边差分
        actual_h_plus = jnp.abs(vox_plus[i, j] - v_ij)
        actual_h_minus = jnp.abs(v_ij - vox_minus[i, j])

        keff_plus = k_eff_only(vox_plus, p=p)
        keff_minus = k_eff_only(vox_minus, p=p)

        # 三种情况：
        # 1) 两侧步长均 > 0：中心差分 (keff_plus - keff_minus) / (h_plus + h_minus)
        # 2) 只有正向：前向差分 (keff_plus - keff0) / h_plus
        # 3) 只有负向：后向差分 (keff0 - keff_minus) / h_minus
        keff0 = k_eff_only(vox, p=p)

        both = jnp.logical_and(actual_h_plus > 0, actual_h_minus > 0)
        only_plus = jnp.logical_and(actual_h_plus > 0, jnp.logical_not(both))
        only_minus = jnp.logical_and(actual_h_minus > 0, jnp.logical_not(both))

        grad_center = (keff_plus - keff_minus) / (actual_h_plus + actual_h_minus + 1e-12)
        grad_fwd = (keff_plus - keff0) / (actual_h_plus + 1e-12)
        grad_bwd = (keff0 - keff_minus) / (actual_h_minus + 1e-12)

        g = jnp.where(both, grad_center,
            jnp.where(only_plus, grad_fwd,
                jnp.where(only_minus, grad_bwd, 0.0)))
        return g

    # 向量化 one_fd，按批次处理
    def batch_fd(batch_indices, vox):
        vmap_fun = jax.vmap(lambda idx: one_fd(idx, vox))
        return vmap_fun(batch_indices)

    # JIT 编译批处理
    batch_fd_jit = jax.jit(batch_fd, static_argnames=[])

    grads_list = []
    for start in range(0, N, batch):
        end = min(start + batch, N)
        idx_batch = indices[start:end]
        g_batch = batch_fd_jit(idx_batch, voxel0)
        grads_list.append(np.array(g_batch))
    grads_flat = np.concatenate(grads_list, axis=0)
    grads_fd = grads_flat.reshape((H, W))
    return grads_fd

# --------------------------------------------------------------------------
# 4. 绘图函数（扩展为 4 子图：结构、AD、FD、差值）
# --------------------------------------------------------------------------

def create_and_save_plots(voxel_structure, T_final, grad_ad, grad_fd, k_eff_val, filename='analysis_results.png'):
    """
    使用 Matplotlib 创建并保存包含多个子图的分析结果图像。
    """
    print("\n正在生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    im1 = ax1.imshow(voxel_structure, cmap='gray_r', interpolation='nearest')
    ax1.set_title('Input Voxel Structure')
    fig.colorbar(im1, ax=ax1, ticks=[0, 1], label='Material (1=Structure, 0=Air)')

    ax2 = axes[0, 1]
    im2 = ax2.imshow(T_final, cmap='hot', interpolation='nearest', vmin=T_COLD, vmax=T_HOT)
    ax2.set_title('Final Temperature Distribution')
    fig.colorbar(im2, ax=ax2, label='Temperature (K)')

    ax3 = axes[1, 0]
    vmax_ad = np.max(np.abs(grad_ad))
    im3 = ax3.imshow(grad_ad, cmap='RdBu_r', interpolation='nearest', vmin=-vmax_ad, vmax=vmax_ad)
    ax3.set_title('Gradient of k_eff (AutoDiff)')
    fig.colorbar(im3, ax=ax3, label='Sensitivity (AD)')

    ax4 = axes[1, 1]
    vmax_fd = np.max(np.abs(grad_fd))
    im4 = ax4.imshow(grad_fd, cmap='RdBu_r', interpolation='nearest', vmin=-vmax_fd, vmax=vmax_fd)
    ax4.set_title('Gradient of k_eff (Finite Difference)')
    fig.colorbar(im4, ax=ax4, label='Sensitivity (FD)')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle(f'Thermal Analysis Results (k_eff = {k_eff_val:.6f})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(filename)
    print(f"✅ 图表已保存至: {filename}")
    plt.show()
    plt.close(fig)

    # 差值单独输出一张图，便于更细观测
    diff = grad_ad - grad_fd
    vmax_diff = np.max(np.abs(diff))
    plt.figure(figsize=(8,6))
    plt.imshow(diff, cmap='RdBu_r', interpolation='nearest', vmin=-vmax_diff, vmax=vmax_diff)
    plt.title('Difference (AD - FD)')
    plt.colorbar(label='Difference')
    plt.xticks([]); plt.yticks([])
    diff_name = filename.replace('.png', '_diff.png')
    plt.tight_layout()
    plt.savefig(diff_name)
    print(f"✅ 差值图已保存至: {diff_name}")
    plt.show()
    plt.close()

# --------------------------------------------------------------------------
# 5. 主执行函数
# --------------------------------------------------------------------------
def analyze_structure(filepath='2d_voxel.npy', p=3.0, fd_eps=FD_EPS, fd_batch=128):
    """
    加载结构，计算 k_eff、自动微分梯度、有限差分梯度，并生成可视化图表。
    """
    print(f"--- 开始分析文件: {filepath} ---")

    try:
        voxel_structure = np.load(filepath)
        print(f"成功加载结构，尺寸: {voxel_structure.shape}")
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。正在创建并使用示例文件。")
        width, height = 30, 30
        structure = np.zeros((height, width), dtype=np.float32)
        structure[10:height-10, width//2 - 5 : width//2 + 5] = 1.0
        structure[10:20, 20:width-20] = 1.0
        structure[height-20:height-10, 20:width-20] = 1.0
        np.save(filepath, structure)
        voxel_structure = structure
        print(f"示例文件 '{filepath}' 已创建。")

    # 确保是 float32
    voxel_structure = voxel_structure.astype(np.float32)
    voxel_structure_jax = jnp.array(voxel_structure, dtype=jnp.float32)

    print("\n正在计算 k_eff 的自动微分梯度和最终温度场...")

    # 自动微分
    def keff_with_aux(vox):
        return calculate_keff_and_temp(vox, p=p)

    value_and_grad_fn = jax.value_and_grad(keff_with_aux, has_aux=True)
    value_and_grad_fn_jit = jax.jit(value_and_grad_fn)

    t0 = time.time()
    ((k_eff_val, T_final), grad_k_eff_ad) = value_and_grad_fn_jit(voxel_structure_jax)
    # block_until_ready
    k_eff_val.block_until_ready()
    T_final.block_until_ready()
    grad_k_eff_ad.block_until_ready()
    t1 = time.time()
    print(f"自动微分完成。耗时: {t1 - t0:.4f} 秒")
    print(f"等效导热系数 (k_eff): {float(k_eff_val):.6f}")

    # 有限差分
    print("\n正在计算有限差分梯度（可能较慢）...")
    t2 = time.time()
    grad_k_eff_fd = finite_difference_gradient(voxel_structure, p=p, eps=fd_eps, batch=fd_batch)
    t3 = time.time()
    print(f"有限差分完成。耗时: {t3 - t2:.4f} 秒")

    # 转为 numpy
    T_final_np = np.array(T_final)
    grad_k_eff_ad_np = np.array(grad_k_eff_ad)
    grad_k_eff_fd_np = np.array(grad_k_eff_fd)

    # 绘图
    create_and_save_plots(voxel_structure, T_final_np, grad_k_eff_ad_np, grad_k_eff_fd_np, float(k_eff_val))
    
    # 额外打印误差度量
    diff = grad_k_eff_ad_np - grad_k_eff_fd_np
    l2 = np.linalg.norm(diff.ravel()) / (np.linalg.norm(grad_k_eff_fd_np.ravel()) + 1e-12)
    linf = np.max(np.abs(diff))
    print(f"相对 L2 误差: {l2:.6e}, L-inf 误差: {linf:.6e}")

    print("--- 分析结束 ---")


if __name__ == '__main__':
    analyze_structure(filepath='2d_voxel.npy', p=3.0, fd_eps=1e-3, fd_batch=128)