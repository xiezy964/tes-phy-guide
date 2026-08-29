import os
import numpy as np
import glob
import math
import cv2
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

# ==========================================
# 全局精度设置 (GPU 训练)
# ==========================================
# 启用 float32 精度矩阵乘法，避免在 GPU 上自动降精度
jax.config.update("jax_default_matmul_precision", "float32")
# 可选：启用 x64 精度（如果需要更高精度，取消注释下一行）
# jax.config.update("jax_enable_x64", True)


# ==========================================
# VP-SDE 类
# ==========================================
class VPSDE:
    """
    Variance Preserving SDE (VP-SDE)
    
    前向 SDE: dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
    
    其中 beta(t) = beta_min + t * (beta_max - beta_min) 是线性调度
    
    解析解:
        x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps
        
    其中:
        alpha(t) = 1 - beta(t)
        alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        """
        Args:
            beta_min: 最小 beta 值
            beta_max: 最大 beta 值  
            T: 终止时间
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        """线性 beta 调度: beta(t) = beta_min + t * (beta_max - beta_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def integral_beta(self, t):
        """beta(s) 从 0 到 t 的积分: integral_0^t beta(s) ds"""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
    
    def alpha_bar(self, t):
        """
        alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)
        这是从 0 到 t 的累积"保留系数"
        """
        return jnp.exp(-0.5 * self.integral_beta(t))
    
    def marginal_prob(self, x_0, t):
        """
        计算边缘分布 p(x_t | x_0) 的均值和标准差
        
        x_t = mean_coef * x_0 + std * eps, eps ~ N(0, I)
        
        Returns:
            mean_coef: sqrt(alpha_bar(t))
            std: sqrt(1 - alpha_bar(t))
        """
        alpha_bar_t = self.alpha_bar(t)
        mean_coef = jnp.sqrt(alpha_bar_t)
        std = jnp.sqrt(1.0 - alpha_bar_t)
        return mean_coef, std
    
    def prior_sampling(self, key, shape):
        """从先验分布 p(x_T) = N(0, I) 采样"""
        return random.normal(key, shape)
    
    def sde_coefficients(self, t):
        """
        SDE 系数: dx = f(x,t) dt + g(t) dw
        
        对于 VP-SDE:
            f(x, t) = -0.5 * beta(t) * x  (drift)
            g(t) = sqrt(beta(t))          (diffusion)
        
        Returns:
            drift_coef: -0.5 * beta(t)
            diffusion: sqrt(beta(t))
        """
        beta_t = self.beta(t)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        return drift_coef, diffusion
    
    def reverse_sde_coefficients(self, x, t, score):
        """
        逆向 SDE 系数:
        dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dw_reverse
        
        其中 score = ∇_x log p_t(x)
        
        对于 VP-SDE:
            f(x, t) = -0.5 * beta(t) * x
            g(t) = sqrt(beta(t))
        
        Returns:
            drift: f(x,t) - g(t)^2 * score
            diffusion: g(t)
        """
        beta_t = self.beta(t)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        
        # 逆向 drift = f(x,t) - g^2 * score
        reverse_drift = drift_coef * x - beta_t * score
        return reverse_drift, diffusion


class DataLoader:
    def __init__(self, data_dir, batch_size, img_size, max_samples=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.file_paths = glob.glob(os.path.join(self.data_dir, "*.npy"))
        if not self.file_paths:
            raise ValueError(f"在 '{self.data_dir}' 目录中未找到任何 .npy 文件。")
        
        if max_samples is not None and max_samples < len(self.file_paths):
            # 随机打乱并截取指定数量的文件
            np.random.shuffle(self.file_paths)
            self.file_paths = self.file_paths[:max_samples]
            print(f"DataLoader: 已随机选取 {len(self.file_paths)} 个样本进行训练。")
    
    def __len__(self):
        return math.ceil(len(self.file_paths) / self.batch_size)
    
    def __iter__(self, key=None):
        file_paths = self.file_paths.copy()
        if key is None:
            np.random.shuffle(file_paths)
        else:
            # 用 key 控制 numpy 的随机性
            np_seed = int(key[0]) if isinstance(key, jnp.ndarray) else int(key)
            rng = np.random.default_rng(np_seed)
            rng.shuffle(file_paths)
        num_skipped = 0
        for i in range(0, len(file_paths), self.batch_size):
            batch_paths = file_paths[i:i+self.batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    img = np.load(path).astype(np.float32)
                    # Check for NaN or Inf in loaded data
                    if np.isnan(img).any() or np.isinf(img).any():
                        print(f"警告: 文件 {path} 包含 NaN 或 Inf，已跳过。")
                        num_skipped += 1
                        continue
                    if img.max() > 1.0:
                        img = img / 255.0
                    # Clip to valid range
                    img = np.clip(img, 0.0, 1.0)
                    if img.shape != (self.img_size, self.img_size):
                        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                    img = np.expand_dims(img, axis=-1)
                    img = img * 2.0 - 1.0
                    batch_images.append(img)
                except Exception as e:
                    print(f"警告: 加载或处理文件 {path} 时出错: {e}，已跳过。")
                    num_skipped += 1
            if not batch_images:
                continue
            yield jnp.array(batch_images)
        if num_skipped > 0:
            print(f"本轮共跳过 {num_skipped} 个损坏或异常图片。")


def setup_dummy_data(data_dir="data", num_files=105, img_size=32):
    """创建一个包含虚拟.npy文件的目录"""
    print(f"正在创建虚拟数据到 '{data_dir}'...")
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_files):
        img = np.random.randint(0, 2, (img_size, img_size)).astype(np.float32)
        np.save(os.path.join(data_dir, f"image_{i}.npy"), img)
    print(f"创建了 {num_files} 个虚拟数据文件。")


# ==========================================
# VP-SDE 训练步骤
# ==========================================
@partial(jax.jit, static_argnums=(3, 4))
def train_step(state, batch, key, model, sde):
    """
    VP-SDE 训练步骤 - 使用 score matching 损失
    
    目标: 训练模型预测噪声 eps (等价于 score function)
    score(x_t, t) = -eps / std(t)
    
    损失函数: E[||eps_pred - eps||^2]
    """
    key, noise_key, t_key = random.split(key, 3)
    batch_size = batch.shape[0]
    
    # 1. 均匀采样时间 t ~ U(eps, 1), 避免 t=0 处的数值问题
    eps_time = 1e-5
    t = random.uniform(t_key, (batch_size,), minval=eps_time, maxval=1.0)
    
    # 2. 采样噪声 eps ~ N(0, I)
    eps = random.normal(noise_key, batch.shape)
    
    # 3. 计算 x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
    mean_coef, std = sde.marginal_prob(batch, t)
    # 广播维度 [B] -> [B, 1, 1, 1]
    mean_coef = mean_coef[:, None, None, None]
    std = std[:, None, None, None]
    x_t = mean_coef * batch + std * eps
    
    # 4. 计算损失
    def loss_fn(params):
        # 模型预测噪声
        eps_pred = model.apply({'params': params}, x_t, t)
        # MSE 损失
        loss = jnp.mean((eps_pred - eps) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # 检查梯度是否有效
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads
    )
    
    state = state.apply_gradients(grads=grads)
    return state, loss, key


# ==========================================
# VP-SDE 采样 (带随机噪声的 SDE 采样)
# ==========================================
@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def sample_sde(key, num_steps, img_size, state, model, sde):
    """
    使用逆向 SDE 进行采样 (Euler-Maruyama 方法)
    
    逆向 SDE: dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dw
    
    对于 VP-SDE:
        score(x,t) ≈ -eps_pred / std(t)
    
    离散化 (从 t 到 t - dt):
        x_{t-dt} = x_t - [f(x,t) - g^2 * score] * dt + g * sqrt(dt) * z
        其中 z ~ N(0, I)
    """
    # 初始化: x_T ~ N(0, I)
    key, init_key = random.split(key)
    x_t = sde.prior_sampling(init_key, (1, img_size, img_size, 1))
    
    # 时间步从 1 -> eps (接近 0)
    eps_time = 1e-5
    timesteps = jnp.linspace(1.0, eps_time, num_steps + 1)
    
    def sde_step(carry, i):
        x_t, key = carry
        key, noise_key = random.split(key)
        
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_curr - t_next  # dt > 0 因为从大 t 到小 t
        
        # 计算 score = -eps_pred / std
        t_array = jnp.array([t_curr])
        eps_pred = model.apply({'params': state.params}, x_t, t_array)
        
        _, std = sde.marginal_prob(x_t, t_curr)
        # 避免除零
        std = jnp.maximum(std, 1e-6)
        score = -eps_pred / std
        
        # SDE 系数
        beta_t = sde.beta(t_curr)
        drift_coef = -0.5 * beta_t
        diffusion = jnp.sqrt(beta_t)
        
        # 逆向 drift
        reverse_drift = drift_coef * x_t - beta_t * score
        
        # Euler-Maruyama 更新 (注意这里是往回走，所以 dt 是正的)
        # x_{t-dt} = x_t - reverse_drift * dt + diffusion * sqrt(dt) * z
        z = random.normal(noise_key, x_t.shape)
        
        # 在最后一步不加噪声
        noise_scale = jnp.where(i < num_steps - 1, 1.0, 0.0)
        x_next = x_t - reverse_drift * dt + noise_scale * diffusion * jnp.sqrt(dt) * z
        
        return (x_next, key), None
    
    # 使用 scan 进行迭代
    (x_0, _), _ = jax.lax.scan(sde_step, (x_t, key), jnp.arange(num_steps))
    
    # 后处理
    generated_img = x_0[0]
    generated_img = (generated_img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    return generated_img


@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def sample_euler(key, num_steps, img_size, state, model, sde):
    """
    使用概率流 ODE 进行采样 (确定性，无噪声)
    
    ODE: dx/dt = f(x,t) - 0.5 * g(t)^2 * score(x,t)
    
    这是 SDE 对应的概率流 ODE，生成相同的边缘分布
    """
    # 初始化: x_T ~ N(0, I)
    key, init_key = random.split(key)
    x_t = sde.prior_sampling(init_key, (1, img_size, img_size, 1))
    
    # 时间步从 1 -> eps
    eps_time = 1e-5
    timesteps = jnp.linspace(1.0, eps_time, num_steps + 1)
    
    def ode_step(i, x_t):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_curr - t_next  # dt > 0
        
        # 计算 score
        t_array = jnp.array([t_curr])
        eps_pred = model.apply({'params': state.params}, x_t, t_array)
        
        _, std = sde.marginal_prob(x_t, t_curr)
        std = jnp.maximum(std, 1e-6)
        score = -eps_pred / std
        
        # ODE 系数
        beta_t = sde.beta(t_curr)
        drift_coef = -0.5 * beta_t
        
        # 概率流 ODE drift = f - 0.5 * g^2 * score
        ode_drift = drift_coef * x_t - 0.5 * beta_t * score
        
        # Euler 更新
        x_next = x_t - ode_drift * dt
        return x_next
    
    x_0 = jax.lax.fori_loop(0, num_steps, ode_step, x_t)
    
    # 后处理
    generated_img = x_0[0]
    generated_img = (generated_img + 1.0) / 2.0
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    return generated_img


@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def sample_ddpm(key, num_steps, img_size, state, model, sde):
    """
    DDPM 风格的采样器 (ancestral sampling)
    
    这是经典 DDPM 论文中的采样方法，使用预定义的离散时间步
    """
    # 初始化
    key, init_key = random.split(key)
    x_t = sde.prior_sampling(init_key, (1, img_size, img_size, 1))
    
    # 离散时间步 (DDPM 风格，均匀间隔)
    timesteps = jnp.linspace(1.0, 0.0, num_steps + 1)
    
    def ddpm_step(carry, i):
        x_t, key = carry
        key, noise_key = random.split(key)
        
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        # 预测噪声
        t_array = jnp.array([t])
        eps_pred = model.apply({'params': state.params}, x_t, t_array)
        
        # 计算 DDPM 系数
        alpha_bar_t = sde.alpha_bar(t)
        alpha_bar_t_prev = sde.alpha_bar(t_prev)
        
        # 避免数值问题
        alpha_bar_t = jnp.maximum(alpha_bar_t, 1e-6)
        alpha_bar_t_prev = jnp.maximum(alpha_bar_t_prev, 1e-6)
        
        # alpha_t = alpha_bar_t / alpha_bar_{t-1}
        alpha_t = alpha_bar_t / alpha_bar_t_prev
        
        # beta_t = 1 - alpha_t
        beta_t = 1.0 - alpha_t
        
        # DDPM 更新公式
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred) + sigma_t * z
        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        
        mean = coef1 * (x_t - coef2 * eps_pred)
        
        # 方差选择: sigma_t^2 = beta_t (简化版)
        # 或者 sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) (更精确)
        sigma_t = jnp.sqrt(beta_t)
        
        # 采样噪声 (最后一步不加)
        z = random.normal(noise_key, x_t.shape)
        noise_scale = jnp.where(i < num_steps - 1, 1.0, 0.0)
        
        x_prev = mean + noise_scale * sigma_t * z
        
        return (x_prev, key), None
    
    (x_0, _), _ = jax.lax.scan(ddpm_step, (x_t, key), jnp.arange(num_steps))
    
    # 后处理
    generated_img = x_0[0]
    generated_img = (generated_img + 1.0) / 2.0
    generated_img = jnp.clip(generated_img, 0.0, 1.0)
    return generated_img


