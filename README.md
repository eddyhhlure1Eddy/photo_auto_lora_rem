# YOLO人脸与人体提取工具

这是一个基于YOLOv3目标检测的图像处理工具集，用于从图像中提取人脸和人体。该项目包含多个实用工具，可以帮助用户进行批量图像处理、人脸提取、人体提取和图像切割等操作。

## 主要功能

### 1. 主程序 (mainGO.py)
- 图像处理主界面
- 人脸和人体检测与提取
- 批量处理文件夹中的图像
- 图像切割功能
- 多种调整选项（尺寸、边距、置信度等）

### 2. 人脸提取工具 (takeface.py)
- 使用YOLOv3进行人脸和上半身检测
- 批量处理图像
- 支持调整边距、最小检测尺寸等参数

### 3. 文本清理工具 (clear_text.py)
- 用于清理指定目录中的所有.txt文件
- 支持递归清理子文件夹

## 环境要求

- Python 3.6+
- OpenCV
- PyQt5
- NumPy
- face_recognition
- dlib

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/yolo-face-body-extractor.git
cd yolo-face-body-extractor
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载YOLOv3模型文件：
   - 如果`models`目录中没有模型文件，请运行`download_models.py`脚本下载所需模型
   - 或手动下载并放置以下文件到`models`目录：
     - yolov3.cfg
     - yolov3.weights

## 使用方法

### 主程序

```bash
python mainGO.py
```

这将启动主界面，您可以通过GUI进行以下操作：
- 加载图像并检测人脸/人体
- 设置各种参数（最小检测尺寸、边距比例等）
- 批量处理整个文件夹中的图像
- 使用图像切割工具

### 人脸提取工具

```bash
python takeface.py
```

这将启动人脸提取工具，您可以：
- 选择输入和输出目录
- 设置检测参数
- 批量提取人脸或上半身

### 文本清理工具

```bash
python clear_text.py
```

这将启动文本清理工具，您可以：
- 选择要清理的目录
- 删除目录及其子目录中的所有.txt文件

## 注意事项

- 该工具使用YOLOv3模型进行检测，请确保您的计算机性能足够运行深度学习模型
- 处理大量或高分辨率图像时可能需要较长时间
- 请备份重要数据，以防意外删除

## 许可证

本项目使用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件 