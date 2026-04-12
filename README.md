# PINet_cpx_cl

本项目是一个基于 PyTorch 的 PINet 实验代码仓库，包含模型定义、数据集处理、损失函数和训练/测试相关脚本。

## 目录说明

- `src/`：核心代码目录，包含模型、数据、损失和工具函数
- `requirements.txt`：项目依赖
- `notebooks/`：实验和验证 Notebook
- `datasets/`：原始/处理后数据集（不建议上传至 GitHub）
- `real_data/`：真实数据集文件（不建议上传至 GitHub）
- `save_dir/`：模型权重和训练结果（不建议上传至 GitHub）
- `result_data_experiment/`：实验结果数据（不建议上传至 GitHub）

## 整理建议

本仓库中较大的实验数据、模型权重、结果文件不建议直接上传到 GitHub，已通过 `.gitignore` 忽略以下类型内容：

- `datasets/`
- `save_dir/`
- `model_saved_pinet_compared*/`
- `pinet_dataset_compared*/`
- `pinet_1600/`
- `real_data/`
- `result_data_experiment/`
- `*.pth` / `*.pt`

如果需要共享模型或数据，建议：

- 使用云存储或第三方下载链接
- 在 README 中写明下载方式
- 保留一个小型示例数据集作为演示

## 使用说明

### 1. 初始化 Git 仓库

```bash
git init
git add .
git commit -m "Initial project structure"
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行训练或测试

请根据 Notebook 或脚本中的注释运行相关命令。建议先使用小规模数据验证环境是否正常。

## 目录优化建议

如果后续想进一步整理，可以考虑：

- 把核心代码移动到 `src/` 或 `pinet/`
- 把 Notebook 放到 `notebooks/`
- 把下载数据脚本、数据说明放到 `data/`

## 备注

当前项目中已有很多实验结果和数据目录，上传 GitHub 时只保留代码与说明文件可以使仓库更干净、更易维护。