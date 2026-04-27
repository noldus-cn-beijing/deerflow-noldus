# 塔式工作站 Linux 配置要求

## 硬件信息

- 设备类型：塔式工作站（带 NVIDIA GPU）
- 团队人数：多人开发，每人独立环境

---

## 1. 操作系统

**推荐：Ubuntu 24.04 LTS Server**（或 Ubuntu 22.04 LTS）

原因：CUDA 兼容性最好，NVIDIA 官方驱动支持最及时，Python/Node.js 生态工具链最顺畅。

---

## 2. 用户与权限

### 2.1 每个开发者一个独立 Linux 账户

| 用户名 | 全名 | 备注 |
|--------|------|------|
| (请提供名单) | | 加入 `developers` 组 |

### 2.2 组结构

```bash
groupadd developers    # 所有开发者
groupadd gpu           # GPU 访问权限
```

- 所有开发者加入 `developers` 组和 `gpu` 组
- 每个用户 home 目录权限 750（`chmod 750 /home/username`）

### 2.3 共享目录（分层设计，避免并发冲突）

**原则：模型权重只读共享，每人实验产出独立隔离。**

```bash
# ── 只读共享：模型基础权重（Qwen3-8B 等，~20GB，不宜每人存一份）
mkdir -p /data/models
chown root:developers /data/models
chmod 755 /data/models          # 所有人可读，只有 root 可写

# ── 每人独立工作目录：微调 checkpoint、实验数据
for user in user1 user2; do
    mkdir -p /data/users/$user
    chown $user:developers /data/users/$user
    chmod 750 /data/users/$user
done
```

**为什么这么设计：**
- `/data/models/` 放 HuggingFace 下载的基础权重，755 只读避免误删/覆盖
- `/data/users/<name>/` 每人一个独立写目录，微调产出互不干扰
- **不在共享目录里放 git 仓库、配置文件、API Key** — 这些各人 clone 自己的 repo

---

## 3. 基础开发工具

```bash
apt-get update && apt-get install -y \
  build-essential \
  curl wget git \
  ca-certificates gnupg \
  htop iotop nvtop \
  tmux \
  vim
```

---

## 4. GPU / CUDA 环境

### 4.1 NVIDIA 驱动

```bash
# 确认 GPU 型号后安装对应驱动，推荐 550 或更新
ubuntu-drivers devices
apt-get install -y nvidia-driver-550
```

### 4.2 CUDA Toolkit 12.x

```bash
# 推荐 CUDA 12.4+（兼容 PyTorch 2.x）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-4
```

### 4.3 cuDNN

```bash
# 从 NVIDIA Developer 下载对应 CUDA 12.x 的 cuDNN 8.9+
apt-get install -y libcudnn8 libcudnn8-dev
```

### 4.4 验证 GPU

```bash
nvidia-smi
nvcc --version
```

---

## 5. Python 环境

### 5.1 Python 版本（系统级）

```bash
add-apt-repository ppa:deadsnakes/ppa -y
apt-get install -y python3.10 python3.12 python3.10-venv python3.12-venv
```

### 5.2 uv 包管理器（全局安装）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv 安装到 /root/.local/bin/uv，对每个用户也加到 PATH
```

### 5.3 每个开发者配置（用户级，各人账号下）

```bash
# uv 会自动管理 venv，项目本身已有 .venv，无需手动创建
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

---

## 6. Node.js 环境

### 6.1 Node.js 22 LTS（系统级）

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
```

### 6.2 pnpm（各用户独立安装）

```bash
# 各用户在自己的账号下运行：
corepack enable
corepack install -g pnpm@10.26.2
```

---

## 7. Docker（可选但推荐）

```bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker <各用户名>
systemctl enable docker

# NVIDIA Container Toolkit（GPU 容器支持）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

---

## 8. 网络与安全

### 8.1 防火墙

```bash
ufw allow 22/tcp          # SSH
ufw allow 2026/tcp        # EthoInsight 开发环境入口（Nginx）
ufw allow 3000/tcp        # Next.js 前端 dev server
ufw allow 8001/tcp        # Gateway API
ufw allow 2024/tcp        # LangGraph Server
ufw enable
```

### 8.2 SSH 配置

```bash
# /etc/ssh/sshd_config 修改：
#   PasswordAuthentication no  （仅密钥登录，如需要）
#   或保留密码登录（内网环境）
```

### 8.3 注意

- 项目配置文件 `config.yaml` 中包含 API Key（NewAPI），建议各开发者在本地 `.env` 或自己的 `config.yaml` 中管理，不要共享
- 建议 `/data/shared` 只放数据集和模型权重，不放含密钥的配置文件

---

## 9. 额外建议

### 9.1 Jupyter Lab（可选，用于数据探索）

```bash
# 各用户自行安装，或系统级安装
pip install jupyterlab
```

### 9.2 fail2ban（安全防护）

```bash
apt-get install -y fail2ban
systemctl enable fail2ban
```

### 9.3 定时自动更新（安全补丁）

```bash
apt-get install -y unattended-upgrades
dpkg-reconfigure unattended-upgrades
```

### 9.4 监控 GPU 占用

`nvtop` 已在上方安装，建议技术人员演示如何使用：
```bash
nvtop          # 实时 GPU 占用
nvidia-smi -l  # 持续刷新 GPU 状态
```

---

## 10. 安装后验证清单

请技术人员完成后逐项确认：

- [ ] 每个开发者可 SSH 登录自己的账号
- [ ] `nvidia-smi` 正常工作，GPU 可见
- [ ] `python3.12 -c "import sys; print(sys.version)"` 正常
- [ ] `uv --version` 正常
- [ ] `node --version` 显示 v22.x
- [ ] `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` 正常（如安装了 Docker）
- [ ] 各用户之间无法访问对方的 home 目录
- [ ] `/data/models` 所有开发者可读（只读）
- [ ] `/data/users/<name>` 仅本人和 root 可读写
- [ ] 各用户 clone 项目后可以 `cd packages/agent/backend && uv sync` 正常安装依赖

---

## 附：项目技术栈速查

| 组件 | 版本要求 |
|------|----------|
| Python | 3.10（ethoinsight 库），3.12（agent 后端） |
| Node.js | 22 |
| 包管理器 | uv（Python），pnpm@10.26.2（Node.js） |
| 前端框架 | Next.js 16 |
| 后端框架 | LangGraph + FastAPI + uvicorn |
| GPU 框架 | PyTorch 2.x + CUDA 12.4（用于 LLM 微调） |
| 反向代理 | Nginx |
| 容器 | Docker（可选） |
