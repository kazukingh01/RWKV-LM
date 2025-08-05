
Ubuntu 24.04 on WSL2

```bash
# 0) 前提：GPU ドライバが入り、nvidia-smi が動くことを確認
nvidia-smi

# 1) 依存ツール
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# 2) GPG キーを配置
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo tee /etc/apt/keyrings/nvidia-container-toolkit.asc

# 3) リポジトリを追加（*stable* ブランチを amd64 用に指定）
echo \
"deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.asc] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" \
| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 4) パッケージリスト更新 & インストール
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 5) Docker 側に nvidia ランタイムを自動設定
sudo nvidia-ctk runtime configure --runtime=docker

# 6) Docker 再起動
sudo systemctl restart docker
```

```bash
sudo docker pull nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
sudo docker run -itd --gpus all --shm-size=16g --name dev -v /home/share:/home/share nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 /bin/bash --login
sudo docker exec -it dev /bin/bash
```

```bash
nvidia-smi
# Fri Aug  1 10:04:28 2025
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 565.77.01              Driver Version: 566.36         CUDA Version: 12.7     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 4090 ...    On  |   00000000:01:00.0 Off |                  N/A |
# | N/A   47C    P0             27W /  175W |       0MiB /  16376MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+

# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |  No running processes found                                                             |
# +-----------------------------------------------------------------------------------------+
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Oct_29_23:50:19_PDT_2024
# Cuda compilation tools, release 12.6, V12.6.85
# Build cuda_12.6.r12.6/compiler.35059454_0
```

python

```bash
apt-get update
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git iputils-ping net-tools vim cron rsyslog
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv/plugins/python-build
bash ./install.sh
INSTALL_PYTHON_VERSION="3.12.10"
/usr/local/bin/python-build -v ${INSTALL_PYTHON_VERSION} ~/local/python-${INSTALL_PYTHON_VERSION}
echo 'export PATH="$HOME/local/python-'${INSTALL_PYTHON_VERSION}'/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

```bash
cd /home/share/10.git/RWKV-LM/RWKV-v7/train_temp/
python -m venv venv && source venv/bin/activate
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu126
pip install pytorch-lightning==1.9.5 deepspeed wandb ninja --upgrade
pip install -r getdata/requirements.txt

mkdir -p data
wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin

sh ./demo-training-prepare.sh
sh ./demo-training-run.sh
```