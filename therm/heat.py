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
K_STRUCTURE = 100.0      # 结构材料的导热系数
K_AIR = 0.1             # 空气（或基底）的导热系数
T_HOT = 100.0           # 上边界温度
T_COLD = 0.0            # 下边界温度

# 求解器参数
MAX_ITERATIONS = 20000
CONVERGENCE_TOL = 1e-6  # 收敛阈值

# --------------------------------------------------------------------------
# 2. 核心计算函数 (使用 fori_loop + jnp.where 实现可微的提前停止)
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['p'])
def calculate_keff_and_temp(voxel_structure, p=3.0):
    """
    计算给定结构的等效导热系数 k_eff 和最终温度场 T_final。
    使用 jax.lax.fori_loop 和 jnp.where 实现一个可微分的、能记录收敛步数的求解器。

    Args:
        voxel_structure (jnp.array): 代表结构的二维数组 (0 for air, 1 for structure)。
        p (float): SIMP惩罚因子。

    Returns:
        tuple: (k_eff, (T_final, final_iteration_count))
            - k_eff (float): 计算出的等效导热系数。
            - T_final (jnp.array): 最终的稳态温度场。
            - final_iteration_count (int): 求解器停止时的迭代步数。
    """
    height, width = voxel_structure.shape
    
    # --- 步骤 A: 计算导热系数场 k_map ---
    k_map = K_AIR + (voxel_structure**p) * (K_STRUCTURE - K_AIR)

    # --- 步骤 B: 使用 fori_loop 迭代求解温度场 T ---
    T_init = jnp.full_like(k_map, (T_HOT + T_COLD) / 2.0)
    T_init = T_init.at[0, :].set(T_HOT)
    T_init = T_init.at[-1, :].set(T_COLD)

    # ==================== 代码修正部分: 使用 fori_loop + jnp.where ====================
    
    # 1. 定义 fori_loop 的循环体函数
    def body_fun(i, state):
        T, _, has_converged, iter_converged = state

        # pad左右（axis=1）模拟绝热边界
        T_pad = jnp.pad(T, ((0, 0), (1, 1)), mode='edge')
        k_pad = jnp.pad(k_map, ((0, 0), (1, 1)), mode='edge')

        # 计算 x方向导热系数
        k_x_fwd = 2 * k_map * k_pad[:, 2:] / (k_map + k_pad[:, 2:])
        k_x_bwd = 2 * k_map * k_pad[:, :-2] / (k_map + k_pad[:, :-2])

        # y方向（依然用jnp.roll，因为上下是定温边界）
        k_y_fwd = 2 * k_map * jnp.roll(k_map, -1, axis=0) / (k_map + jnp.roll(k_map, -1, axis=0))
        k_y_bwd = jnp.roll(k_y_fwd, 1, axis=0)

        numerator = (
            k_y_bwd * jnp.roll(T, 1, axis=0) +
            k_y_fwd * jnp.roll(T, -1, axis=0) +
            k_x_bwd * T_pad[:, :-2] +
            k_x_fwd * T_pad[:, 2:]
        )
        denominator = k_y_bwd + k_y_fwd + k_x_bwd + k_x_fwd

        T_candidate = numerator / denominator

        # 上下定温
        T_candidate = T_candidate.at[0, :].set(T_HOT)
        T_candidate = T_candidate.at[-1, :].set(T_COLD)

        error = jnp.max(jnp.abs(T_candidate - T))
        should_update = jnp.logical_not(has_converged)
        T_new = jnp.where(should_update, T_candidate, T)
        newly_converged = (error < CONVERGENCE_TOL)
        next_has_converged = jnp.logical_or(has_converged, newly_converged)
        iter_converged_new = jnp.where(
            jnp.logical_and(should_update, newly_converged),
            i,
            iter_converged
        )

        return (T_new, T, next_has_converged, iter_converged_new)

    # 2. 设置初始状态并运行 fori_loop
    # 初始状态: (当前温度, 上一轮温度, 是否已收敛, 收敛步数)
    # has_converged 初始为 False
    # iter_converged 初始为一个大数，代表还未收敛
    initial_state = (T_init, T_init, jnp.array(False), jnp.array(MAX_ITERATIONS))
    
    # fori_loop 返回的是执行 MAX_ITERATIONS 次后的最终状态
    T_final, _, _, final_iteration_count = jax.lax.fori_loop(0, MAX_ITERATIONS, body_fun, initial_state)
    # ======================================================================
    
    # --- 步骤 C: 计算总热流 Q_total ---
    k_interface_top = 2 * k_map[0, :] * k_map[1, :] / (k_map[0, :] + k_map[1, :])
    q_y = k_interface_top * (T_final[0, :] - T_final[1, :])
    Q_total = jnp.sum(q_y)

    # --- 步骤 D: 计算等效导热系数 k_eff ---
    delta_T = T_HOT - T_COLD
    L = height
    A = width
    k_eff = (Q_total * L) / (A * delta_T)
    
    # 将 T_final 和 final_iteration_count 作为辅助输出返回
    return k_eff, (T_final, final_iteration_count)

# --------------------------------------------------------------------------
# 3. 绘图函数 (无变化)
# --------------------------------------------------------------------------

def create_and_save_plots(voxel_structure, T_final, grad_k_eff, k_eff_val, filename='analysis_results.png'):
    """
    使用 Matplotlib 创建并保存包含多个子图的分析结果图像。
    """
    print("\n正在生成可视化图表...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1 = axes[0]
    im1 = ax1.imshow(voxel_structure, cmap='gray_r', interpolation='nearest')
    ax1.set_title('Input Voxel Structure')
    fig.colorbar(im1, ax=ax1, ticks=[0, 1], label='Material (1=Structure, 0=Air)')

    ax2 = axes[1]
    im2 = ax2.imshow(T_final, cmap='RdBu_r', interpolation='nearest', vmin=T_COLD, vmax=T_HOT)
    ax2.set_title('Final Temperature Distribution')
    fig.colorbar(im2, ax=ax2, label='Temperature (K)')

    ax3 = axes[2]
    vmax_abs = np.max(np.abs(grad_k_eff))
    im3 = ax3.imshow(grad_k_eff, cmap='RdBu_r', interpolation='nearest', vmin=-vmax_abs, vmax=vmax_abs)
    ax3.set_title('Gradient of k_eff (Sensitivity)')
    fig.colorbar(im3, ax=ax3, label='Sensitivity')

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle(f'Thermal Analysis Results (Calculated k_eff = {k_eff_val:.4f})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(filename)
    print(f"✅ 图表已保存至: {filename}")
    plt.show()
    plt.close(fig)

# --------------------------------------------------------------------------
# 4. 主执行函数
# --------------------------------------------------------------------------
def analyze_structure(filepath='therm/2d_voxel.npy'):
    """
    加载结构，计算 k_eff 和梯度，并生成可视化图表。
    """
    print(f"--- 开始分析文件: {filepath} ---")

    try:
        voxel_structure = np.load(filepath)
        print(f"成功加载结构，尺寸: {voxel_structure.shape}")
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。正在创建并使用示例文件。")
        width, height = 100, 100
        structure = np.zeros((height, width), dtype=np.float32)
        structure[10:height-10, width//2 - 5 : width//2 + 5] = 1.0
        structure[10:20, 20:width-20] = 1.0
        structure[height-20:height-10, 20:width-20] = 1.0
        np.save(filepath, structure)
        voxel_structure = structure
        print(f"示例文件 '{filepath}' 已创建。")

    voxel_structure_jax = jnp.array(voxel_structure, dtype=jnp.float32)

    print("\n正在计算 k_eff 的梯度和最终温度场...")
    
    value_and_grad_fn = jax.value_and_grad(calculate_keff_and_temp, has_aux=True)
    value_and_grad_fn_jit = jax.jit(value_and_grad_fn)

    start_time = time.time()
    
    ((k_eff_val, (T_final, final_iter)), grad_k_eff) = value_and_grad_fn_jit(voxel_structure_jax)
    
    # 确保计算完成，以便我们能访问到 final_iter 的真实值
    final_iter.block_until_ready()
    k_eff_val.block_until_ready()
    end_time = time.time()

    print("求解器信息:")
    iter_val = int(final_iter)
    
    # 检查循环是因为收敛停止还是因为达到最大迭代次数
    if iter_val < MAX_ITERATIONS:
        print(f"✅ 求解器在第 {iter_val} 次迭代时逻辑收敛。")
    else:
        print(f"⚠️ 求解器在达到最大迭代次数 {MAX_ITERATIONS} 后停止，可能未完全收敛。")
    
    print(f"梯度和温度场计算完成。耗时: {end_time - start_time:.4f} 秒")
    print(f"等效导热系数 (k_eff): {k_eff_val:.4f}")

    T_final_np = np.array(T_final)
    grad_k_eff_np = np.array(grad_k_eff)

    create_and_save_plots(voxel_structure, T_final_np, grad_k_eff_np, k_eff_val)
    
    print("--- 分析结束 ---")


if __name__ == '__main__':
    analyze_structure(filepath='therm/2d_voxel.npy')