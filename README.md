# 工人工地安全检测系统

## 项目简介

本项目基于 YOLOv11 实现工地工人安全帽佩戴检测，支持数据集自动划分、模型训练、测试与 Web 可视化推理。

---

## 目录结构

```text
├── app.py                # Streamlit Web 检测界面
├── train.py              # 训练脚本
├── test.py               # 推理与评估脚本
├── dataset_split.py      # 数据集划分与VOC转YOLO（txt）脚本
├── dataset.yaml          # 数据集配置文件
├── runs/
│   └── detect/
|       └── trainX/valX/predictX    # 训练/评估/推理结果
├── datasets/
│   └── hard_hat_workers/ # 数据集
│       ├── images/       # 原始图片
│       ├── annotations/  # VOC格式xml标注
│       ├── train/val/test/
│           ├── images/  # 划分后图片
│           ├── labels/  # YOLO格式标签
```

---

## 环境依赖

- Python 3.10
- [pytorch 2.7.1](https://pytorch.org/get-started/locally/#windows-installation) (cuda 12.6)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [streamlit](https://streamlit.io/)
- moviepy

### 安装方式

```bash
# 创建虚拟环境
conda create -n yolo11 python=3.10
conda activate yolo11

# 安装pytorch（需根据CUDA版本选择）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install ultralytics streamlit moviepy

# 或者直接使用requirements.txt
pip install -r requirements.txt
```

---

## 数据集准备与划分

1. 数据集链接：
   - [https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)
   - [https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
2. 将数据集的原始图片放入 `datasets/hard_hat_workers/images/`，VOC标注xml放入 `datasets/hard_hat_workers/annotations/`
3. 编辑 `dataset.yaml`
4. 运行数据集划分与格式转换脚本：

```bash
python dataset_split.py
```

脚本会自动将图片和标注划分为 train/val/test，并转换为YOLO格式。

---

## 训练与测试

- 训练模型：（须先下载预训练权重，如[yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)）

```bash
python train.py
```

- 推理与验证：
  
`test.py` 支持图片、视频、文件夹推理和数据集评估，需通过命令行参数指定模式和路径。

- **推理（单张图片/视频/文件夹）**：

```bash
python test.py --mode infer --path 路径 [--weights 权重文件路径]
```

示例：

```bash
python test.py --mode infer --path datasets/test_image.jpg
python test.py --mode infer --path datasets/test_video.mp4
python test.py --mode infer --path datasets/hard_hat_workers/images/
```

- **数据集评估（在测试集上评估模型）**：

```bash
python test.py --mode val [--data 数据集配置yaml] [--weights 权重文件路径]
```

示例：

```bash
python test.py --mode val --data dataset.yaml
```

参数说明：

- `--mode`：选择 `infer`（推理）或 `val`（评估）
- `--path`：推理模式下必填，指定图片/视频/文件夹路径
- `--weights`：模型权重文件路径，默认 `./runs/detect/train2/weights/best.pt`
- `--data`：数据集配置 yaml，验证模式下可选，默认 `dataset.yaml`

---

## Web端可视化检测

```bash
streamlit run app.py
```

- 运行后在web页面上传图片即可检测，结果会显示在页面上，并统计检测数量
