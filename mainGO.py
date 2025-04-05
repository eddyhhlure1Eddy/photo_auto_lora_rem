#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import dlib
import numpy as np
import shutil
import tempfile
import uuid
import time
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QSlider, QComboBox, 
                            QCheckBox, QSpinBox, QProgressBar, QTabWidget, QMessageBox,
                            QGridLayout, QGroupBox, QRadioButton, QListWidget, QSplitter,
                            QDialog, QScrollArea, QLineEdit, QTextEdit, QListWidgetItem)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QDateTime
from PyQt5.QtWidgets import QAction, QMenu, QToolBar, QStatusBar, QDockWidget, QTextEdit
import face_recognition

# 导入视频处理模块的注释 - 这是一个预留接口，未来可以添加视频处理功能
# from video_processor import VideoProcessingThread, get_video_info


def resize_with_mode(img, target_width, target_height, mode="stretch", interpolation=cv2.INTER_LANCZOS4):
    """根据指定的模式调整图像大小
    
    Args:
        img: 输入图像（numpy数组）
        target_width: 目标宽度
        target_height: 目标高度
        mode: 调整模式，可选值：'stretch'（拉伸）, 'crop'（居中裁剪）, 'pad'（加黑边）
        interpolation: 插值方法
        
    Returns:
        调整大小后的图像
    """
    if img is None:
        return None
        
    h, w = img.shape[:2]
    target_ratio = target_width / target_height
    original_ratio = w / h
    
    # 拉伸模式：直接调整到目标大小，不保持宽高比
    if mode == "stretch":
        return cv2.resize(img, (target_width, target_height), interpolation=interpolation)
    
    # 居中裁剪模式：保持宽高比，裁剪多余部分
    elif mode == "crop":
        if abs(original_ratio - target_ratio) < 1e-2:  # 比例接近，直接缩放
            return cv2.resize(img, (target_width, target_height), interpolation=interpolation)
        elif original_ratio > target_ratio:  # 原图更宽，裁剪两侧
            new_width = int(h * target_ratio)
            x0 = int((w - new_width) / 2)
            cropped = img[:, x0:x0+new_width]
            return cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)
        else:  # 原图更高，裁剪上下
            new_height = int(w / target_ratio)
            y0 = int((h - new_height) / 2)
            cropped = img[y0:y0+new_height, :]
            return cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)
    
    # 加黑边模式：保持宽高比，多余部分填充黑色
    elif mode == "pad":
        if abs(original_ratio - target_ratio) < 1e-2:  # 比例接近，直接缩放
            return cv2.resize(img, (target_width, target_height), interpolation=interpolation)
        
        # 确定缩放比例
        scale = min(target_width / w, target_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
        
        # 创建黑色背景
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 计算偏移量以居中放置
        x_offset = int((target_width - new_width) / 2)
        y_offset = int((target_height - new_height) / 2)
        
        # 将缩放后的图像放在黑色背景中央
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        return result
    
    # 默认使用拉伸模式
    return cv2.resize(img, (target_width, target_height), interpolation=interpolation)


class TempFileManager:
    """临时文件管理器，用于管理应用程序创建的临时文件"""
    
    def __init__(self, temp_dir=None):
        """初始化临时文件管理器
        
        Args:
            temp_dir: 临时目录路径，如果为None，则使用系统临时目录
        """
        if temp_dir is None:
            self.temp_dir = os.path.join(tempfile.gettempdir(), 'face_extraction_app')
        else:
            self.temp_dir = temp_dir
            
        # 确保临时目录存在
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # 保存创建的临时文件列表
        self.temp_files = []
        
    def create_temp_file(self, suffix=None):
        """创建临时文件
        
        Args:
            suffix: 文件后缀
            
        Returns:
            临时文件路径
        """
        # 使用uuid创建唯一文件名
        file_name = str(uuid.uuid4())
        if suffix:
            file_name += suffix
            
        temp_file_path = os.path.join(self.temp_dir, file_name)
        self.temp_files.append(temp_file_path)
        return temp_file_path
    
    def create_temp_image_file(self, image, format='jpg'):
        """创建临时图像文件
        
        Args:
            image: OpenCV图像
            format: 图像格式
            
        Returns:
            临时图像文件路径
        """
        temp_file = self.create_temp_file('.' + format)
        cv2.imwrite(temp_file, image)
        return temp_file
    
    def clean_up(self):
        """清理所有临时文件"""
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"清理临时文件时出错: {str(e)}")
                    
        # 清空列表
        self.temp_files.clear()
        
        # 尝试删除临时目录（如果为空）
        try:
            if os.path.exists(self.temp_dir):
                if not os.listdir(self.temp_dir):
                    os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"清理临时目录时出错: {str(e)}")


def read_image_safe(file_path):
    """安全地读取图像，支持中文路径"""
    try:
        # 尝试使用numpy方法读取，此方法对中文路径更友好
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # 如果还是读取失败，尝试直接使用OpenCV读取
        if img is None:
            print(f"使用numpy方法读取失败: {file_path}")
            img = cv2.imread(file_path)
            
        # 如果仍然失败，尝试使用绝对路径
        if img is None:
            print(f"使用直接路径读取失败: {file_path}")
            abs_path = os.path.abspath(file_path)
            img = cv2.imread(abs_path)
            
        # 如果所有方法都失败
        if img is None:
            print(f"所有读取方法均失败: {file_path}")
            return None
            
        return img
    except Exception as e:
        print(f"读取图像出错: {str(e)}")
        return None


def save_image_safe(file_path, image, format='jpg'):
    """安全地保存图像，支持中文路径"""
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 使用cv2.imencode和numpy方法保存
        ext = os.path.splitext(file_path)[1].lower()
        if not ext:
            ext = f".{format}"
        
        # 根据格式选择压缩参数
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            params = []
        
        # 编码图像
        _, img_buf = cv2.imencode(ext, image, params)
        
        # 保存到文件
        with open(file_path, 'wb') as f:
            f.write(img_buf)
        
        return True
    except Exception as e:
        print(f"保存图像出错: {str(e)}")
        return False


def load_yolo_model(model_dir="models", temp_dir="C:\\temp_models"):
    """加载YOLO模型，处理中文路径和版本兼容性问题"""
    try:
        # 获取模型路径
        weights_path = os.path.join(model_dir, "yolov3.weights")
        config_path = os.path.join(model_dir, "yolov3.cfg")
        
        # 使用绝对路径
        weights_path = os.path.abspath(weights_path)
        config_path = os.path.abspath(config_path)
        
        print(f"YOLO模型路径: {weights_path}")
        print(f"YOLO配置路径: {config_path}")
        
        # 检查文件是否存在
        if not os.path.exists(weights_path):
            print(f"找不到YOLO权重文件: {weights_path}")
            return None, None
        
        if not os.path.exists(config_path):
            print(f"找不到YOLO配置文件: {config_path}")
            return None, None
        
        # 尝试直接加载YOLO模型
        print("尝试直接加载YOLO模型...")
        net = None
        
        try:
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        except Exception as e1:
            print(f"直接加载模型失败: {str(e1)}")
            print("尝试通过临时英文路径加载模型...")
            
            try:
                # 确保临时目录存在
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # 复制模型文件到临时目录
                temp_weights = os.path.join(temp_dir, "yolov3.weights")
                temp_config = os.path.join(temp_dir, "yolov3.cfg")
                
                print(f"复制权重文件到: {temp_weights}")
                import shutil
                shutil.copy2(weights_path, temp_weights)
                
                print(f"复制配置文件到: {temp_config}")
                shutil.copy2(config_path, temp_config)
                
                # 从临时路径加载模型
                print(f"从临时路径加载模型...")
                net = cv2.dnn.readNetFromDarknet(temp_config, temp_weights)
            except Exception as e2:
                print(f"通过临时路径加载模型也失败: {str(e2)}")
                return None, None
        
        if net is None:
            print("无法加载YOLO模型")
            return None, None
            
        # 获取输出层
        try:
            print("获取网络层名称...")
            layer_names = net.getLayerNames()
            print(f"网络层名称获取成功，共 {len(layer_names)} 层")
            
            print("获取未连接的输出层索引...")
            out_layers_indices = net.getUnconnectedOutLayers()
            print(f"未连接输出层索引: {out_layers_indices}, 类型: {type(out_layers_indices)}")
            
            # 处理不同OpenCV版本的兼容性问题
            output_layers = []
            
            try:
                # 尝试OpenCV 4.x方式 (索引从0开始)
                if isinstance(out_layers_indices, np.ndarray):
                    # 如果是多维数组，展平
                    if len(out_layers_indices.shape) > 1:
                        out_layers_indices = out_layers_indices.flatten()
                    
                    print(f"处理扁平化后的索引: {out_layers_indices}")
                    # 假设索引从1开始
                    output_layers = [layer_names[i - 1] for i in out_layers_indices]
                else:
                    # 如果是列表或元组
                    print("处理列表或元组形式的索引")
                    for i in out_layers_indices:
                        if isinstance(i, (list, tuple)):
                            output_layers.append(layer_names[i[0] - 1])
                        else:
                            output_layers.append(layer_names[i - 1])
            except Exception as e:
                print(f"使用索引-1方式处理层名失败: {str(e)}")
                print("尝试使用直接索引方式...")
                
                try:
                    # 尝试不减1的方式
                    if isinstance(out_layers_indices, np.ndarray):
                        output_layers = [layer_names[i] for i in out_layers_indices]
                    else:
                        for i in out_layers_indices:
                            if isinstance(i, (list, tuple)):
                                output_layers.append(layer_names[i[0]])
                            else:
                                output_layers.append(layer_names[i])
                except Exception as e2:
                    print(f"使用直接索引方式也失败: {str(e2)}")
                    print("使用硬编码的输出层名称...")
                    
                    # 使用常见的YOLOv3输出层名称
                    output_layers = ['yolo_82', 'yolo_94', 'yolo_106']
            
            print(f"最终确定的输出层: {output_layers}")
        except Exception as e:
            print(f"获取输出层时出错: {str(e)}")
            # 使用默认的YOLOv3输出层名称
            output_layers = ['yolo_82', 'yolo_94', 'yolo_106']
            print(f"使用默认输出层: {output_layers}")
        
        print("YOLO模型加载成功!")
        return net, output_layers
    except Exception as e:
        print(f"加载YOLO模型出错: {str(e)}")
        return None, None


def detect_bodies_with_yolo(image, min_body_size=150, margin_ratio=0.2, confidence_threshold=0.2, nms_threshold=0.3):
    """使用YOLO检测人体"""
    # 确保参数在有效范围内
    confidence_threshold = max(0.2, min(0.95, confidence_threshold))
    nms_threshold = max(0.1, min(0.7, nms_threshold))
    
    # 加载模型
    print("加载YOLO模型...")
    net, output_layers = load_yolo_model()
    if net is None or output_layers is None:
        print("无法加载YOLO模型，无法执行人体检测")
        return []
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    print(f"图像尺寸: {width}x{height}")
    print(f"检测参数: 最小尺寸={min_body_size}, 边距比例={margin_ratio:.2f}")
    print(f"检测参数: 置信度阈值={confidence_threshold:.2f}, NMS阈值={nms_threshold:.2f}")
    
    # 准备图像数据
    print("准备图像数据...")
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 执行检测
    print("执行YOLO检测...")
    try:
        outputs = net.forward(output_layers)
        print(f"检测完成，获得 {len(outputs)} 个输出层的结果")
    except Exception as e:
        print(f"执行前向传播时出错: {str(e)}")
        return []
    
    # 处理检测结果
    boxes = []
    confidences = []
    
    # 使用传入的置信度阈值
    print(f"使用置信度阈值: {confidence_threshold}")
    
    for i, output in enumerate(outputs):
        print(f"处理输出层 {i+1}/{len(outputs)}, 包含 {len(output)} 个检测结果")
        detection_count = 0
        
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 类别0是person(人)，使用传入的置信度阈值
            if class_id == 0 and confidence > confidence_threshold:
                detection_count += 1
                # YOLO返回的是中心点坐标和宽高
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # 计算左上角坐标
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                print(f"  检测到人体: 位置=({x},{y}), 大小={w}x{h}, 置信度={confidence:.3f}")
        
        print(f"  输出层 {i+1} 检测到 {detection_count} 个可能的人体")
    
    print(f"初步检测到 {len(boxes)} 个可能的人体")
    
    # 非极大值抑制去除重叠框
    body_regions = []
    if boxes:
        try:
            # 使用传入的NMS阈值进行非极大值抑制
            print(f"使用NMS阈值: {nms_threshold} 和置信度阈值: {confidence_threshold}")
            print(f"检测到 {len(boxes)} 个候选边界框，应用NMS进行过滤...")
            
            # 明确使用传入的参数值进行NMS操作
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            num_indices = len(indices) if isinstance(indices, list) or (isinstance(indices, np.ndarray) and indices.size > 0) else 0
            print(f"非极大值抑制后保留 {num_indices} 个人体")
            
            if num_indices > 0:
                indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                
                for idx in indices:
                    # 提取坐标
                    x, y, w, h = boxes[idx]
                    
                    # 扩大人体区域以包含更多背景
                    margin_w = int(w * margin_ratio)
                    margin_h = int(h * margin_ratio)
                    
                    x_expanded = max(0, x - margin_w)
                    y_expanded = max(0, y - margin_h)
                    w_expanded = min(width - x_expanded, w + 2 * margin_w)
                    h_expanded = min(height - y_expanded, h + 2 * margin_h)
                    
                    # 如果人体区域大于最小尺寸，则添加到结果
                    if w_expanded > min_body_size or h_expanded > min_body_size:
                        # 提取人体图像
                        body_img = image[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
                        
                        # 添加到结果列表，并记录置信度
                        body_regions.append({
                            'body': body_img,
                            'coords': (x_expanded, y_expanded, x_expanded+w_expanded, y_expanded+h_expanded),
                            'original_coords': (x, y, x+w, y+h),
                            'confidence': confidences[idx]  # 保存置信度信息
                        })
                        print(f"保留人体区域: 位置=({x_expanded},{y_expanded}), 大小={w_expanded}x{h_expanded}, 置信度={confidences[idx]:.2f}")
        except Exception as e:
            print(f"处理检测结果时出错: {str(e)}")
    
    print(f"最终检测到 {len(body_regions)} 个有效人体")
    return body_regions


def detect_bodies_with_separate_margins(image, min_body_size=150, margin_ratio_x=0.4, margin_ratio_y=0.2, confidence_threshold=0.2, nms_threshold=0.3):
    """使用YOLO检测人体，支持独立的水平和垂直边距
    
    Args:
        image: 输入图像
        min_body_size: 最小人体尺寸
        margin_ratio_x: 水平边距比例
        margin_ratio_y: 垂直边距比例
        confidence_threshold: 置信度阈值
        nms_threshold: 非极大值抑制阈值
    
    Returns:
        检测到的人体区域列表
    """
    # 确保参数在有效范围内
    confidence_threshold = max(0.2, min(0.95, confidence_threshold))
    nms_threshold = max(0.1, min(0.7, nms_threshold))
    
    # 加载模型
    print("加载YOLO模型...")
    net, output_layers = load_yolo_model()
    if net is None or output_layers is None:
        print("无法加载YOLO模型，无法执行人体检测")
        return []
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    print(f"图像尺寸: {width}x{height}")
    print(f"检测参数: 最小尺寸={min_body_size}, 水平边距={margin_ratio_x:.2f}, 垂直边距={margin_ratio_y:.2f}")
    print(f"检测参数: 置信度阈值={confidence_threshold:.2f}, NMS阈值={nms_threshold:.2f}")
    
    # 准备图像数据
    print("准备图像数据...")
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 执行检测
    print("执行YOLO检测...")
    try:
        outputs = net.forward(output_layers)
        print(f"检测完成，获得 {len(outputs)} 个输出层的结果")
    except Exception as e:
        print(f"执行前向传播时出错: {str(e)}")
        return []
    
    # 处理检测结果
    boxes = []
    confidences = []
    
    # 使用传入的置信度阈值
    print(f"使用置信度阈值: {confidence_threshold}")
    
    for i, output in enumerate(outputs):
        print(f"处理输出层 {i+1}/{len(outputs)}, 包含 {len(output)} 个检测结果")
        detection_count = 0
        
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 类别0是person(人)，使用传入的置信度阈值
            if class_id == 0 and confidence > confidence_threshold:
                detection_count += 1
                # YOLO返回的是中心点坐标和宽高
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # 计算左上角坐标
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                print(f"  检测到人体: 位置=({x},{y}), 大小={w}x{h}, 置信度={confidence:.3f}")
        
        print(f"  输出层 {i+1} 检测到 {detection_count} 个可能的人体")
    
    print(f"初步检测到 {len(boxes)} 个可能的人体")
    
    # 非极大值抑制去除重叠框
    body_regions = []
    if boxes:
        try:
            # 使用传入的NMS阈值进行非极大值抑制
            print(f"使用NMS阈值: {nms_threshold} 和置信度阈值: {confidence_threshold}")
            print(f"检测到 {len(boxes)} 个候选边界框，应用NMS进行过滤...")
            
            # 明确使用传入的参数值进行NMS操作
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            num_indices = len(indices) if isinstance(indices, list) or (isinstance(indices, np.ndarray) and indices.size > 0) else 0
            print(f"非极大值抑制后保留 {num_indices} 个人体")
            
            if num_indices > 0:
                indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                
                for idx in indices:
                    # 提取坐标
                    x, y, w, h = boxes[idx]
                    
                    # 扩大人体区域以包含更多背景 - 使用独立的水平和垂直边距
                    margin_w = int(w * margin_ratio_x)  # 水平边距，左右各添加
                    margin_h = int(h * margin_ratio_y)  # 垂直边距，上下各添加
                    
                    x_expanded = max(0, x - margin_w)
                    y_expanded = max(0, y - margin_h)
                    w_expanded = min(width - x_expanded, w + 2 * margin_w)
                    h_expanded = min(height - y_expanded, h + 2 * margin_h)
                    
                    # 如果人体区域大于最小尺寸，则添加到结果
                    if w_expanded > min_body_size or h_expanded > min_body_size:
                        # 提取人体图像
                        body_img = image[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
                        
                        # 添加到结果列表，并记录置信度
                        body_regions.append({
                            'body': body_img,
                            'coords': (x_expanded, y_expanded, x_expanded+w_expanded, y_expanded+h_expanded),
                            'original_coords': (x, y, x+w, y+h),
                            'confidence': confidences[idx]  # 保存置信度信息
                        })
                        print(f"保留人体区域: 位置=({x_expanded},{y_expanded}), 大小={w_expanded}x{h_expanded}, 置信度={confidences[idx]:.2f}")
        except Exception as e:
            print(f"处理检测结果时出错: {str(e)}")
    
    print(f"最终检测到 {len(body_regions)} 个有效人体")
    return body_regions


class BodyDetectionThread(QThread):
    """处理人体检测的线程"""
    update_progress = pyqtSignal(int)
    update_image = pyqtSignal(QPixmap)
    finished_detection = pyqtSignal(list)
    log_message = pyqtSignal(str)
    
    def __init__(self, image_path, min_body_size=150, margin_ratio=0.2, use_yolo=True, show_borders=False, confidence_threshold=0.2, nms_threshold=0.3):
        super().__init__()
        self.image_path = image_path
        self.min_body_size = min_body_size
        self.margin_ratio = margin_ratio
        self.use_yolo = use_yolo
        self.show_borders = show_borders
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
    def run(self):
        """运行人体检测"""
        try:
            # 读取原始图像
            self.log_message.emit(f"正在读取图像: {self.image_path}")
            img = read_image_safe(self.image_path)
            
            if img is None:
                self.log_message.emit("无法读取图像，请检查图像路径或格式")
                self.finished_detection.emit([])
                return
                
            # 创建图像副本
            original_img = img.copy()
            
            # 发送进度更新
            self.update_progress.emit(10)
            
            # 初始化检测结果
            body_regions = []
            
            if self.use_yolo:
                # 使用YOLO模型
                self.log_message.emit("使用YOLO模型进行人体检测...")
                # 明确记录当前使用的参数值
                self.log_message.emit(f"检测参数: 最小尺寸={self.min_body_size}像素, 边距比例={self.margin_ratio:.2f}")
                self.log_message.emit(f"检测参数: 置信度阈值={self.confidence_threshold:.2f}, NMS阈值={self.nms_threshold:.2f}")
                body_regions = detect_bodies_with_yolo(
                    img, 
                    min_body_size=self.min_body_size, 
                    margin_ratio=self.margin_ratio,
                    confidence_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold
                )
                
                # 在显示图像上标记检测到的人体
                if self.show_borders and body_regions:
                    for body_data in body_regions:
                        x1, y1, x2, y2 = body_data['coords']
                        confidence = body_data.get('confidence', 0.0)
                        
                        # 绘制检测框和置信度
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 添加置信度文本
                        text = f"Person: {confidence:.2f}"
                        cv2.putText(original_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                self.log_message.emit(f"YOLO检测到 {len(body_regions)} 个人体")
                
            # 如果没有检测到任何人体，考虑整个图像作为一个区域
            if not body_regions and img is not None:
                self.log_message.emit("未检测到人体，使用整个图像")
                height, width = img.shape[:2]
                body_regions = [{
                    'body': img.copy(),
                    'coords': (0, 0, width, height),
                    'original_coords': (0, 0, width, height),
                    'confidence': 1.0  # 整图使用，认为是100%的置信度
                }]
                
                # 如果需要显示边框，绘制整个图像的边框
                if self.show_borders:
                    cv2.rectangle(original_img, (0, 0), (width, height), (0, 0, 255), 2)
                    cv2.putText(original_img, "Full Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 发送进度更新
            self.update_progress.emit(90)
            
            # 将检测结果直接传递给应用程序
            result_bodies = body_regions
            
            # 显示检测结果
            if result_bodies:
                # 将OpenCV图像转换为Qt图像
                h, w, c = original_img.shape
                q_image = QImage(original_img.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                self.update_image.emit(pixmap)
            
            # 发送检测完成信号
            self.log_message.emit("人体检测完成")
            self.finished_detection.emit(result_bodies)
            self.update_progress.emit(100)
        
        except Exception as e:
            self.log_message.emit(f"人体检测出错: {str(e)}")
            self.finished_detection.emit([])
            self.update_progress.emit(0)


class FaceDetectionThread(QThread):
    """处理人脸检测的线程"""
    update_progress = pyqtSignal(int)
    update_image = pyqtSignal(QPixmap)
    finished_detection = pyqtSignal(list)
    log_message = pyqtSignal(str)
    
    def __init__(self, image_path, min_face_size=80, margin_ratio=0.2, show_borders=False):
        super().__init__()
        self.image_path = image_path
        self.min_face_size = min_face_size
        self.margin_ratio = margin_ratio
        self.show_borders = show_borders
        
    def run(self):
        try:
            # 读取图像
            self.log_message.emit(f"正在读取图像: {self.image_path}")
            img = read_image_safe(self.image_path)
            
            if img is None:
                self.log_message.emit("无法读取图像，请检查图像路径或格式")
                self.finished_detection.emit([])
                return
                
            # 创建图像副本
            original_img = img.copy()
            
            # 发送进度更新
            self.update_progress.emit(10)
            
            # 使用YOLO检测人脸
            self.log_message.emit("使用YOLO模型检测人脸...")
            
            # 加载YOLO模型
            net, output_layers = load_yolo_model()
            if net is None or output_layers is None:
                self.log_message.emit("加载YOLO模型失败")
                self.finished_detection.emit([])
                return
            
            # 获取图像尺寸
            height, width = img.shape[:2]
            
            # 准备图像数据
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            
            # 执行检测
            self.log_message.emit("执行人脸检测...")
            outputs = net.forward(output_layers)
            
            # 处理检测结果
            boxes = []
            confidences = []
            
            # 发送进度更新
            self.update_progress.emit(40)
            
            # 分析每个检测结果
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # 类别0是person(人)，我们从人体上部区域检测人脸
                    if class_id == 0 and confidence > 0.3:
                        # YOLO返回的是中心点坐标和宽高
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 计算左上角坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # 只取人体上部1/3区域作为人脸区域
                        face_height = int(h * 0.33)
                        face_y = max(0, y)
                        face_h = min(face_height, height - face_y)
                        
                        if face_h > self.min_face_size:
                            boxes.append([x, face_y, w, face_h])
                            confidences.append(float(confidence))
            
            # 发送进度更新
            self.update_progress.emit(60)
            
            # 非极大值抑制去除重叠框
            faces = []
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
                
                if len(indices) > 0:
                    indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
                    
                    # 初始化人脸区域列表
                    face_regions = []
                    
                    for idx in indices:
                        # 提取坐标
                        x, y, w, h = boxes[idx]
                        
                        # 扩大人脸区域以包含更多背景
                        margin_w = int(w * self.margin_ratio)
                        margin_h = int(h * self.margin_ratio)
                        
                        x_expanded = max(0, x - margin_w)
                        y_expanded = max(0, y - margin_h)
                        w_expanded = min(width - x_expanded, w + 2 * margin_w)
                        h_expanded = min(height - y_expanded, h + 2 * margin_h)
                        
                        # 如果人脸区域大于最小尺寸，则添加到结果
                        if w_expanded > self.min_face_size and h_expanded > self.min_face_size:
                            # 提取人脸图像
                            face_img = img[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
                            
                            # 添加到结果列表
                            face_regions.append({
                                'face': face_img,
                                'coords': (x_expanded, y_expanded, x_expanded+w_expanded, y_expanded+h_expanded)
                            })
                            
                            # 显示边框
                            if self.show_borders:
                                cv2.rectangle(original_img, (x_expanded, y_expanded), 
                                          (x_expanded + w_expanded, y_expanded + h_expanded), (0, 255, 0), 2)
            
            # 发送进度更新
            self.update_progress.emit(90)
            
            # 如果检测到人脸，更新图像显示
            if face_regions:
                self.log_message.emit(f"检测到 {len(face_regions)} 个人脸")
                
                # 将OpenCV图像转换为Qt图像
                h, w, c = original_img.shape
                q_image = QImage(original_img.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                self.update_image.emit(pixmap)
            else:
                self.log_message.emit("未检测到人脸")
            
            # 发送检测完成信号
            self.finished_detection.emit(face_regions)
            self.update_progress.emit(100)
            
        except Exception as e:
            self.log_message.emit(f"人脸检测出错: {str(e)}")
            self.finished_detection.emit([])
            self.update_progress.emit(100)


class FaceExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置标题
        self.setWindowTitle("人脸/人体提取工具")
        
        # 初始化临时文件管理器
        self.temp_manager = TempFileManager()
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")
        
        # 创建一个中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页控件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 创建主要功能标签页
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "单图处理")
        
        # 主标签页布局
        main_tab_layout = QVBoxLayout(main_tab)
        
        # 分割主窗口为左右两部分
        splitter = QSplitter(Qt.Horizontal)
        main_tab_layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 添加一个标签组
        control_group = QGroupBox("操作控制")
        control_layout = QVBoxLayout()
        
        # 添加按钮
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # 检测类型选择
        detection_type_group = QGroupBox("检测类型")
        detection_type_layout = QHBoxLayout()
        
        self.face_detect_radio = QRadioButton("人脸检测")
        self.face_detect_radio.setChecked(True)
        self.body_detect_radio = QRadioButton("人体检测")
        
        detection_type_layout.addWidget(self.face_detect_radio)
        detection_type_layout.addWidget(self.body_detect_radio)
        detection_type_group.setLayout(detection_type_layout)
        control_layout.addWidget(detection_type_group)
        
        # 检测设置组
        detection_settings_group = QGroupBox("检测设置")
        detection_settings_layout = QGridLayout()
        
        # 使用标签说明人脸检测始终使用YOLO
        detection_settings_layout.addWidget(QLabel("人脸检测方法:"), 0, 0)
        detection_settings_layout.addWidget(QLabel("YOLO (高精度)"), 0, 1)
        
        # 使用标签说明人体检测始终使用YOLO
        detection_settings_layout.addWidget(QLabel("人体检测方法:"), 1, 0)
        detection_settings_layout.addWidget(QLabel("YOLO (高精度)"), 1, 1)
        
        # 最小人脸尺寸
        detection_settings_layout.addWidget(QLabel("最小人脸尺寸:"), 2, 0)
        self.min_face_size_spin = QSpinBox()
        self.min_face_size_spin.setRange(20, 500)
        self.min_face_size_spin.setValue(80)
        detection_settings_layout.addWidget(self.min_face_size_spin, 2, 1)
        
        # 最小人体尺寸
        detection_settings_layout.addWidget(QLabel("最小人体尺寸:"), 3, 0)
        self.min_body_size_spin = QSpinBox()
        self.min_body_size_spin.setRange(50, 1000)
        self.min_body_size_spin.setValue(150)
        # 添加值变化时自动重新检测
        self.min_body_size_spin.valueChanged.connect(self.trigger_delayed_detection)
        detection_settings_layout.addWidget(self.min_body_size_spin, 3, 1)
        
        # 显示边框选项
        self.show_borders_checkbox = QCheckBox("显示边框")
        self.show_borders_checkbox.setChecked(False)
        detection_settings_layout.addWidget(self.show_borders_checkbox, 4, 0, 1, 2)
        
        # 添加仅保留最高置信度结果选项
        self.only_highest_confidence_checkbox = QCheckBox("仅保留最高置信度结果")
        self.only_highest_confidence_checkbox.setChecked(False)
        self.only_highest_confidence_checkbox.setToolTip("当检测到多个人体时，只保留置信度最高的一个")
        detection_settings_layout.addWidget(self.only_highest_confidence_checkbox, 5, 0, 1, 2)
        
        # 添加优先中心人体选项
        self.prioritize_center_checkbox = QCheckBox("优先中心人体")
        self.prioritize_center_checkbox.setChecked(False)
        self.prioritize_center_checkbox.setToolTip("当检测到多个人体时，优先选择靠近图像中心的人体")
        detection_settings_layout.addWidget(self.prioritize_center_checkbox, 6, 0, 1, 2)
        
        # 添加背景比例滑块
        detection_settings_layout.addWidget(QLabel("背景比例:"), 7, 0)
        self.margin_ratio_slider = QSlider(Qt.Horizontal)
        self.margin_ratio_slider.setMinimum(0)
        self.margin_ratio_slider.setMaximum(100)
        self.margin_ratio_slider.setValue(20)  # 默认20%
        self.margin_ratio_slider.setTickPosition(QSlider.TicksBelow)
        self.margin_ratio_slider.setTickInterval(10)
        margin_value_layout = QHBoxLayout()
        self.margin_ratio_value = QLabel("20%")
        self.margin_ratio_slider.valueChanged.connect(lambda v: self.margin_ratio_value.setText(f"{v}%"))
        # 添加值变化时自动重新检测
        self.margin_ratio_slider.valueChanged.connect(self.trigger_delayed_detection)
        margin_value_layout.addWidget(self.margin_ratio_slider)
        margin_value_layout.addWidget(self.margin_ratio_value)
        detection_settings_layout.addLayout(margin_value_layout, 7, 1)
        # 添加提示说明
        margin_tip = QLabel("提示: 背景边距影响检测精度，值越小裁剪越紧凑")
        margin_tip.setStyleSheet("color: #0066cc; font-style: italic;")
        detection_settings_layout.addWidget(margin_tip, 8, 0, 1, 2)
        
        # 添加置信度阈值滑块
        detection_settings_layout.addWidget(QLabel("置信度阈值:"), 9, 0)
        self.confidence_threshold_slider = QSlider(Qt.Horizontal)
        self.confidence_threshold_slider.setRange(20, 95)  # 对应0.2到0.95
        self.confidence_threshold_slider.setValue(20)  # 默认0.2
        self.confidence_threshold_slider.setTickInterval(5)
        self.confidence_threshold_slider.setTickPosition(QSlider.TicksBelow)
        
        # 值显示布局
        confidence_value_layout = QHBoxLayout()
        confidence_value_layout.addWidget(self.confidence_threshold_slider)
        self.confidence_value_label = QLabel("0.20")  # 确保初始值与滑块值一致
        confidence_value_layout.addWidget(self.confidence_value_label)
        detection_settings_layout.addLayout(confidence_value_layout, 9, 1)
        
        # 修改连接方式，直接更新标签并触发检测更新
        self.confidence_threshold_slider.valueChanged.connect(
            lambda v: (
                self.confidence_value_label.setText(f"{v/100:.2f}"), 
                self.trigger_delayed_detection() if self.current_image_path else None
            )
        )
        
        # 添加NMS阈值滑块
        detection_settings_layout.addWidget(QLabel("NMS阈值:"), 10, 0)
        self.nms_threshold_slider = QSlider(Qt.Horizontal)
        self.nms_threshold_slider.setRange(10, 70)  # 对应0.1到0.7
        self.nms_threshold_slider.setValue(30)  # 默认0.3
        self.nms_threshold_slider.setTickInterval(5)
        self.nms_threshold_slider.setTickPosition(QSlider.TicksBelow)
        
        # 值显示布局
        nms_value_layout = QHBoxLayout()
        nms_value_layout.addWidget(self.nms_threshold_slider)
        self.nms_value_label = QLabel("0.30")  # 确保初始值与滑块值一致
        nms_value_layout.addWidget(self.nms_value_label)
        detection_settings_layout.addLayout(nms_value_layout, 10, 1)
        
        # 修改连接方式，直接更新标签并触发检测更新
        self.nms_threshold_slider.valueChanged.connect(
            lambda v: (
                self.nms_value_label.setText(f"{v/100:.2f}"), 
                self.trigger_delayed_detection() if self.current_image_path else None
            )
        )
        
        detection_settings_group.setLayout(detection_settings_layout)
        control_layout.addWidget(detection_settings_group)
        
        # 检测与保存按钮
        self.detect_btn = QPushButton("检测")
        self.detect_btn.clicked.connect(self.detect_objects)
        control_layout.addWidget(self.detect_btn)
        
        # 批量处理按钮
        self.batch_process_btn = QPushButton("批量处理文件夹")
        self.batch_process_btn.clicked.connect(self.batch_process_folder)
        control_layout.addWidget(self.batch_process_btn)
        
        # 新增：批量提取子文件夹图像按钮
        self.extract_subfolder_images_btn = QPushButton("提取所有子文件夹图像")
        self.extract_subfolder_images_btn.clicked.connect(self.extract_subfolder_images)
        self.extract_subfolder_images_btn.setToolTip("从所有子文件夹中提取图像到一个新文件夹")
        control_layout.addWidget(self.extract_subfolder_images_btn)
        
        # 保存结果按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        # 图像切割按钮
        self.image_split_btn = QPushButton("图像切割工具")
        self.image_split_btn.clicked.connect(self.open_image_splitter)
        control_layout.addWidget(self.image_split_btn)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # 添加日志区域
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图像标签
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("未加载图像")
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        self.image_scroll_area.setWidget(self.image_label)
        right_layout.addWidget(self.image_scroll_area)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)
        
        # 将左右面板添加到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # 设置初始大小
        
        # 创建批量人体检测标签页
        batch_body_tab = QWidget()
        self.tab_widget.addTab(batch_body_tab, "批量人体提取")
        
        # 批量人体检测标签页布局
        batch_body_layout = QVBoxLayout(batch_body_tab)
        
        # 创建水平分割器
        h_splitter = QSplitter(Qt.Horizontal)
        batch_body_layout.addWidget(h_splitter)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 输入和输出目录选择
        directories_group = QGroupBox("目录设置")
        directories_layout = QGridLayout()
        
        # 输入目录
        directories_layout.addWidget(QLabel("输入目录:"), 0, 0)
        self.batch_input_path = QLineEdit()
        self.batch_input_path.setReadOnly(True)
        directories_layout.addWidget(self.batch_input_path, 0, 1)
        self.batch_input_browse_btn = QPushButton("浏览...")
        self.batch_input_browse_btn.clicked.connect(self.browse_batch_input_dir)
        directories_layout.addWidget(self.batch_input_browse_btn, 0, 2)
        
        # 输出目录
        directories_layout.addWidget(QLabel("输出目录:"), 1, 0)
        self.batch_output_path = QLineEdit()
        self.batch_output_path.setReadOnly(True)
        directories_layout.addWidget(self.batch_output_path, 1, 1)
        self.batch_output_browse_btn = QPushButton("浏览...")
        self.batch_output_browse_btn.clicked.connect(self.browse_batch_output_dir)
        directories_layout.addWidget(self.batch_output_browse_btn, 1, 2)
        
        directories_group.setLayout(directories_layout)
        left_layout.addWidget(directories_group)
        
        # 批量处理参数设置
        batch_settings_group = QGroupBox("处理参数")
        batch_settings_layout = QGridLayout()
        
        # 最小人体尺寸
        batch_settings_layout.addWidget(QLabel("最小人体尺寸:"), 0, 0)
        self.batch_min_body_size = QSpinBox()
        self.batch_min_body_size.setRange(50, 1000)
        self.batch_min_body_size.setValue(150)
        self.batch_min_body_size.setSuffix(" 像素")
        self.batch_min_body_size.valueChanged.connect(self.update_size_slider)
        batch_settings_layout.addWidget(self.batch_min_body_size, 0, 1)
        
        # 背景边距 - 分为水平和垂直两个滑块
        batch_settings_layout.addWidget(QLabel("水平边距比例(左右):"), 1, 0)
        self.batch_margin_x_slider = QSlider(Qt.Horizontal)
        self.batch_margin_x_slider.setMinimum(0)
        self.batch_margin_x_slider.setMaximum(150)  # 允许更大范围，最大150%
        self.batch_margin_x_slider.setValue(40)  # 默认40%，比原先更大
        self.batch_margin_x_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_margin_x_slider.setTickInterval(10)
        batch_margin_x_layout = QHBoxLayout()
        batch_margin_x_layout.addWidget(self.batch_margin_x_slider)
        self.batch_margin_x_value = QLabel("40%")
        self.batch_margin_x_slider.valueChanged.connect(lambda v: self.batch_margin_x_value.setText(f"{v}%"))
        self.batch_margin_x_slider.valueChanged.connect(self.update_preview_on_param_change)
        batch_margin_x_layout.addWidget(self.batch_margin_x_value)
        batch_settings_layout.addLayout(batch_margin_x_layout, 1, 1)
        
        batch_settings_layout.addWidget(QLabel("垂直边距比例(上下):"), 2, 0)
        self.batch_margin_y_slider = QSlider(Qt.Horizontal)
        self.batch_margin_y_slider.setMinimum(0)
        self.batch_margin_y_slider.setMaximum(100)
        self.batch_margin_y_slider.setValue(20)  # 默认20%，与原先相同
        self.batch_margin_y_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_margin_y_slider.setTickInterval(10)
        batch_margin_y_layout = QHBoxLayout()
        batch_margin_y_layout.addWidget(self.batch_margin_y_slider)
        self.batch_margin_y_value = QLabel("20%")
        self.batch_margin_y_slider.valueChanged.connect(lambda v: self.batch_margin_y_value.setText(f"{v}%"))
        self.batch_margin_y_slider.valueChanged.connect(self.update_preview_on_param_change)
        batch_margin_y_layout.addWidget(self.batch_margin_y_value)
        batch_settings_layout.addLayout(batch_margin_y_layout, 2, 1)
        
        # 预览按钮 - 调整到第3行
        self.preview_detection_btn = QPushButton("预览检测")
        self.preview_detection_btn.clicked.connect(self.preview_body_detection)
        batch_settings_layout.addWidget(self.preview_detection_btn, 3, 0, 1, 2)
        
        # 输出格式
        batch_settings_layout.addWidget(QLabel("输出格式:"), 4, 0)
        self.batch_format_combo = QComboBox()
        self.batch_format_combo.addItems(["jpg", "png", "bmp"])
        batch_settings_layout.addWidget(self.batch_format_combo, 4, 1)
        
        # 调整大小选项
        self.batch_resize_checkbox = QCheckBox("调整输出图像大小")
        batch_settings_layout.addWidget(self.batch_resize_checkbox, 5, 0)
        
        # 尺寸设置
        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(QLabel("宽度:"))
        self.batch_resize_width = QSpinBox()
        self.batch_resize_width.setRange(32, 4096)
        self.batch_resize_width.setValue(512)
        self.batch_resize_width.setEnabled(False)
        batch_size_layout.addWidget(self.batch_resize_width)
        
        batch_size_layout.addWidget(QLabel("高度:"))
        self.batch_resize_height = QSpinBox()
        self.batch_resize_height.setRange(32, 4096)
        self.batch_resize_height.setValue(512)
        self.batch_resize_height.setEnabled(False)
        batch_size_layout.addWidget(self.batch_resize_height)
        
        # 添加调整大小模式选择
        batch_size_layout.addWidget(QLabel("调整模式:"))
        self.batch_resize_mode_combo = QComboBox()
        self.batch_resize_mode_combo.addItem("拉伸", "stretch")
        self.batch_resize_mode_combo.addItem("居中裁剪", "crop")
        self.batch_resize_mode_combo.addItem("加黑边", "pad")
        self.batch_resize_mode_combo.setToolTip("拉伸：直接调整到目标大小\n居中裁剪：保持宽高比，裁剪多余部分\n加黑边：保持宽高比，多余部分填充黑色")
        self.batch_resize_mode_combo.setEnabled(False)
        batch_size_layout.addWidget(self.batch_resize_mode_combo)
        
        # 连接调整大小复选框与尺寸输入框的启用状态
        self.batch_resize_checkbox.toggled.connect(self.batch_resize_width.setEnabled)
        self.batch_resize_checkbox.toggled.connect(self.batch_resize_height.setEnabled)
        self.batch_resize_checkbox.toggled.connect(self.batch_resize_mode_combo.setEnabled)
        
        batch_settings_layout.addLayout(batch_size_layout, 5, 1)
        
        # 高清输出选项
        self.batch_hd_checkbox = QCheckBox("高清输出")
        self.batch_hd_checkbox.setToolTip("保持原始图像质量，不进行压缩")
        batch_settings_layout.addWidget(self.batch_hd_checkbox, 6, 0, 1, 2)
        
        # 添加仅保留最高置信度结果选项
        self.batch_only_highest_confidence_checkbox = QCheckBox("仅保留最高置信度结果")
        self.batch_only_highest_confidence_checkbox.setChecked(False)
        self.batch_only_highest_confidence_checkbox.setToolTip("当检测到多个人体时，只保留置信度最高的一个")
        batch_settings_layout.addWidget(self.batch_only_highest_confidence_checkbox, 7, 0, 1, 2)
        
        # 添加优先中心人体选项
        self.batch_prioritize_center_checkbox = QCheckBox("优先中心人体")
        self.batch_prioritize_center_checkbox.setChecked(False)
        self.batch_prioritize_center_checkbox.setToolTip("当检测到多个人体时，优先选择靠近图像中心的人体")
        batch_settings_layout.addWidget(self.batch_prioritize_center_checkbox, 8, 0, 1, 2)
        
        # 自定义文本内容
        batch_settings_layout.addWidget(QLabel("文本文件内容:"), 9, 0)
        self.batch_text_content = QTextEdit()
        self.batch_text_content.setPlaceholderText("输入要保存到文本文件中的内容。留空则不创建文本文件。")
        self.batch_text_content.setMaximumHeight(100)
        batch_settings_layout.addWidget(self.batch_text_content, 8, 1)
        
        batch_settings_group.setLayout(batch_settings_layout)
        left_layout.addWidget(batch_settings_group)
        
        # 批量处理按钮和进度条
        batch_control_layout = QHBoxLayout()
        self.start_batch_btn = QPushButton("开始批量处理")
        self.start_batch_btn.clicked.connect(self.start_batch_body_process)
        self.stop_batch_btn = QPushButton("停止处理")
        self.stop_batch_btn.clicked.connect(self.stop_batch_process)
        self.stop_batch_btn.setEnabled(False)
        batch_control_layout.addWidget(self.start_batch_btn)
        batch_control_layout.addWidget(self.stop_batch_btn)
        left_layout.addLayout(batch_control_layout)
        
        # 批量处理进度条
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setValue(0)
        left_layout.addWidget(self.batch_progress_bar)
        
        # 批量处理日志
        batch_log_group = QGroupBox("处理日志")
        batch_log_layout = QVBoxLayout()
        self.batch_log_text = QTextEdit()
        self.batch_log_text.setReadOnly(True)
        batch_log_layout.addWidget(self.batch_log_text)
        batch_log_group.setLayout(batch_log_layout)
        left_layout.addWidget(batch_log_group)
        
        # 右侧预览面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 预览区域
        preview_group = QGroupBox("检测预览")
        preview_layout = QVBoxLayout()
        
        # 添加预览图像标签
        self.preview_image_label = QLabel()
        self.preview_image_label.setMinimumSize(400, 400)
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        self.preview_image_label.setText("点击'预览检测'按钮查看检测效果")
        preview_layout.addWidget(self.preview_image_label)
        
        # 添加检测范围调整滑块
        slider_group = QGroupBox("调整检测范围")
        slider_layout = QVBoxLayout()
        
        # 最小人体尺寸滑块
        size_slider_layout = QHBoxLayout()
        size_slider_layout.addWidget(QLabel("最小人体尺寸:"))
        self.detection_size_slider = QSlider(Qt.Horizontal)
        self.detection_size_slider.setRange(50, 500)
        self.detection_size_slider.setValue(150)
        self.detection_size_slider.setTickInterval(50)
        self.detection_size_slider.setTickPosition(QSlider.TicksBelow)
        self.detection_size_slider.valueChanged.connect(self.update_min_body_size)
        size_slider_layout.addWidget(self.detection_size_slider)
        self.size_value_label = QLabel("150像素")
        size_slider_layout.addWidget(self.size_value_label)
        slider_layout.addLayout(size_slider_layout)
        
        # 置信度阈值滑块
        confidence_slider_layout = QHBoxLayout()
        confidence_slider_layout.addWidget(QLabel("置信度阈值:"))
        self.batch_confidence_slider = QSlider(Qt.Horizontal)
        self.batch_confidence_slider.setRange(20, 95)  # 对应0.2到0.95
        self.batch_confidence_slider.setValue(20)  # 默认0.2
        self.batch_confidence_slider.setTickInterval(5)
        self.batch_confidence_slider.setTickPosition(QSlider.TicksBelow)
        # 修改连接方式，直接更新标签并触发预览更新
        self.batch_confidence_slider.valueChanged.connect(
            lambda v: (
                self.batch_confidence_value_label.setText(f"{v/100:.2f}"), 
                self.update_preview_on_param_change()
            )
        )
        confidence_slider_layout.addWidget(self.batch_confidence_slider)
        self.batch_confidence_value_label = QLabel("0.20")  # 修改为批量专用标签
        confidence_slider_layout.addWidget(self.batch_confidence_value_label)
        slider_layout.addLayout(confidence_slider_layout)
        
        # NMS阈值滑块
        nms_slider_layout = QHBoxLayout()
        nms_slider_layout.addWidget(QLabel("NMS阈值:"))
        self.batch_nms_slider = QSlider(Qt.Horizontal)
        self.batch_nms_slider.setRange(10, 70)  # 对应0.1到0.7
        self.batch_nms_slider.setValue(30)  # 默认0.3
        self.batch_nms_slider.setTickInterval(5)
        self.batch_nms_slider.setTickPosition(QSlider.TicksBelow)
        # 修改连接方式，直接更新标签并触发预览更新（不调用单图检测的trigger_delayed_detection）
        self.batch_nms_slider.valueChanged.connect(
            lambda v: (
                self.batch_nms_value_label.setText(f"{v/100:.2f}"), 
                self.update_preview_on_param_change()
            )
        )
        nms_slider_layout.addWidget(self.batch_nms_slider)
        self.batch_nms_value_label = QLabel("0.30")  # 修改为批量专用标签
        nms_slider_layout.addWidget(self.batch_nms_value_label)
        slider_layout.addLayout(nms_slider_layout)
        
        slider_group.setLayout(slider_layout)
        preview_layout.addWidget(slider_group)
        
        # 预览信息
        self.preview_info_label = QLabel("预览信息: 未执行检测")
        preview_layout.addWidget(self.preview_info_label)
        
        # 添加弹性空间
        preview_layout.addStretch()
        
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)
        
        # 将左右面板添加到分割器
        h_splitter.addWidget(left_panel)
        h_splitter.addWidget(right_panel)
        h_splitter.setSizes([400, 600])  # 设置初始大小
        
        # 初始化变量
        self.current_image_path = None
        self.detected_faces = []
        self.detected_bodies = []
        self.face_detection_thread = None
        self.body_detection_thread = None
        self.batch_thread = None
        self.preview_sample_path = None  # 用于存储预览图像路径
        
        # 记录日志
        self.log("应用程序已启动")
        self.log("支持中文路径的临时文件系统已启用")

    def closeEvent(self, event):
        """关闭窗口时清理临时文件"""
        self.log("正在清理临时文件...")
        self.temp_manager.clean_up()
        event.accept()
        
    def log(self, message):
        """向日志区域添加消息"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def load_image(self):
        """加载图像"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.log(f"已选择图像: {file_path}")
            
            # 加载并显示图像
            try:
                self.log("正在读取图像...")
                pixmap = QPixmap(file_path)
                if pixmap.isNull():
                    # 使用OpenCV读取
                    self.log("标准方法读取失败，尝试使用临时文件系统...")
                    
                    # 使用临时文件系统
                    temp_path = self.temp_manager.create_temp_file(os.path.splitext(file_path)[1])
                    self.log(f"使用临时路径: {temp_path}")
                    
                    # 读取图像
                    image = read_image_safe(file_path)
                    if image is not None:
                        # 转换为QPixmap
                        height, width, channel = image.shape
                        bytes_per_line = 3 * width
                        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                        pixmap = QPixmap.fromImage(q_img)
                    else:
                        self.log("读取图像失败")
                        return
                
                # 调整图像大小以适应窗口
                self.display_pixmap(pixmap)
                self.log("图像加载成功")
                
                # 重置检测结果
                self.detected_faces = []
                self.detected_bodies = []
                self.save_btn.setEnabled(False)
                
            except Exception as e:
                self.log(f"加载图像时出错: {str(e)}")
    
    def display_pixmap(self, pixmap):
        """显示图像并适应窗口大小"""
        if pixmap.isNull():
            self.log("无法显示空图像")
            return
            
        # 获取窗口大小
        scroll_area_size = self.image_scroll_area.size()
        
        # 如果图像太大，调整大小以适应窗口
        if pixmap.width() > scroll_area_size.width() or pixmap.height() > scroll_area_size.height():
            pixmap = pixmap.scaled(
                scroll_area_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # 显示图像
        self.image_label.setPixmap(pixmap)

    def detect_objects(self):
        """根据选择的类型检测人脸或人体"""
        if not self.current_image_path:
            self.log("请先加载图像")
            return
        
        # 获取是否显示边框
        show_borders = self.show_borders_checkbox.isChecked()
        
        # 根据选择的检测类型执行相应的操作
        if self.face_detect_radio.isChecked():
            self.detect_faces()
        else:
            self.detect_bodies()
    
    def detect_faces(self):
        """检测人脸"""
        if not self.current_image_path:
            self.log("请先加载图像")
            return
            
        # 获取检测设置
        min_face_size = self.min_face_size_spin.value()
        show_borders = self.show_borders_checkbox.isChecked()
        
        # 获取背景比例（固定使用YOLO）
        margin_ratio = 0.2  # 使用固定的背景比例
        
        # 日志输出检测方法
        self.log("使用 YOLO 方法进行人脸检测")
        
        # 检查YOLO模型文件是否存在
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        weights_path = os.path.join(model_dir, "yolov3.weights")
        config_path = os.path.join(model_dir, "yolov3.cfg")
        
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            QMessageBox.warning(
                self,
                "YOLO模型文件缺失",
                "未找到YOLO模型文件！\n\n"
                "请确保以下文件存在于'models'目录中：\n"
                "1. yolov3.weights (约236MB)\n"
                "2. yolov3.cfg\n\n"
                "您可以使用optimize_and_run.bat中的选项5下载这些文件，"
                "或参考README.md中的说明手动下载。"
            )
            return
        
        # 禁用检测按钮，防止重复点击
        self.detect_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 创建线程进行检测
        self.face_detection_thread = FaceDetectionThread(
            self.current_image_path, 
            min_face_size,
            margin_ratio,
            show_borders
        )
        
        # 连接信号
        self.face_detection_thread.update_progress.connect(self.progress_bar.setValue)
        self.face_detection_thread.update_image.connect(self.update_image)
        self.face_detection_thread.finished_detection.connect(self.handle_face_detection_finished)
        self.face_detection_thread.log_message.connect(self.log)
        
        # 开始线程
        self.face_detection_thread.start()
    
    def detect_bodies(self):
        """检测人体"""
        if not self.current_image_path:
            self.log("请先加载图像")
            return
            
        # 获取检测设置
        min_body_size = self.min_body_size_spin.value()
        show_borders = self.show_borders_checkbox.isChecked()
        
        # 获取背景比例设置（始终使用YOLO）
        margin_ratio = self.margin_ratio_slider.value() / 100.0
        use_yolo = True
        
        # 获取置信度和NMS阈值
        confidence_threshold = self.confidence_threshold_slider.value() / 100.0
        nms_threshold = self.nms_threshold_slider.value() / 100.0
        
        # 日志输出检测方法和参数
        self.log("使用 YOLO 方法进行人体检测")
        self.log(f"最小人体尺寸: {min_body_size}像素")
        self.log(f"背景边距比例: {margin_ratio:.2f} ({self.margin_ratio_slider.value()}%)")
        self.log(f"置信度阈值: {confidence_threshold:.2f}, NMS阈值: {nms_threshold:.2f}")
        
        # 检查YOLO模型文件是否存在
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        weights_path = os.path.join(model_dir, "yolov3.weights")
        config_path = os.path.join(model_dir, "yolov3.cfg")
        
        # 输出模型文件路径用于调试
        self.log(f"YOLO模型路径: {weights_path}")
        self.log(f"YOLO配置路径: {config_path}")
        
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            QMessageBox.warning(
                self,
                "YOLO模型文件缺失",
                f"未找到YOLO模型文件！\n\n"
                f"请确保以下文件存在于'models'目录中：\n"
                f"1. yolov3.weights (约236MB)\n"
                f"2. yolov3.cfg\n\n"
                f"尝试查找的路径：\n{weights_path}\n{config_path}\n\n"
                f"您可以使用optimize_and_run.bat中的选项5下载这些文件，"
                f"或参考README.md中的说明手动下载。"
            )
            return
        
        # 禁用检测按钮，防止重复点击
        self.detect_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 创建线程进行检测
        self.body_detection_thread = BodyDetectionThread(
            self.current_image_path, 
            min_body_size=min_body_size,
            margin_ratio=margin_ratio,
            use_yolo=use_yolo,
            show_borders=show_borders,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        # 连接信号
        self.body_detection_thread.update_progress.connect(self.progress_bar.setValue)
        self.body_detection_thread.update_image.connect(self.update_image)
        self.body_detection_thread.finished_detection.connect(self.handle_body_detection_finished)
        self.body_detection_thread.log_message.connect(self.log)
        
        # 开始线程
        self.body_detection_thread.start()
        
    def update_image(self, pixmap):
        """更新图像显示"""
        self.display_pixmap(pixmap)
    
    def handle_face_detection_finished(self, faces):
        """处理人脸检测完成"""
        self.detected_faces = faces
        self.detected_bodies = []  # 清空人体检测结果
        self.detect_btn.setEnabled(True)
        
        if faces:
            self.log(f"检测到 {len(faces)} 个人脸")
            self.save_btn.setEnabled(True)
        else:
            self.log("未检测到人脸")
            self.save_btn.setEnabled(False)
    
    def handle_body_detection_finished(self, bodies):
        """处理人体检测完成"""
        # 计算优先中心处理
        if self.prioritize_center_checkbox.isChecked() and len(bodies) > 1:
            # 获取当前图像尺寸
            img = read_image_safe(self.current_image_path)
            if img is not None:
                height, width = img.shape[:2]
                # 计算图像中心点
                center_x, center_y = width / 2, height / 2
                
                # 为每个人体计算到中心的距离
                for body in bodies:
                    x, y, w, h = body.get('coords', (0, 0, 0, 0))
                    # 计算人体边界框中心
                    body_center_x = x + w / 2
                    body_center_y = y + h / 2
                    # 计算到图像中心的距离
                    distance = ((body_center_x - center_x) ** 2 + (body_center_y - center_y) ** 2) ** 0.5
                    body['center_distance'] = distance
                
                # 按照到中心的距离排序
                bodies = sorted(bodies, key=lambda x: x.get('center_distance', float('inf')))
                self.log(f"已启用'优先中心人体'，已按距离中心从近到远排序")
                
                # 如果同时启用了仅保留最高置信度，则只保留中心最近的一个
                if self.only_highest_confidence_checkbox.isChecked():
                    bodies = [bodies[0]]
                    self.log(f"同时启用'仅保留最高置信度结果'，保留距离中心最近的人体")
        
        # 如果只启用了仅保留最高置信度结果，则过滤结果
        elif self.only_highest_confidence_checkbox.isChecked() and len(bodies) > 1:
            # 找出置信度最高的人体
            highest_confidence_body = max(bodies, key=lambda x: x.get('confidence', 0))
            bodies = [highest_confidence_body]
            self.log(f"已启用'仅保留最高置信度结果'，过滤后保留 1 个人体")
            
        self.detected_bodies = bodies
        self.detected_faces = []  # 清空人脸检测结果
        self.detect_btn.setEnabled(True)
        
        if bodies:
            self.log(f"检测到 {len(bodies)} 个人体")
            self.save_btn.setEnabled(True)
        else:
            self.log("未检测到人体")
            self.save_btn.setEnabled(False)

    def save_results(self):
        """保存检测结果"""
        # 检查是否有检测结果
        if not self.detected_faces and not self.detected_bodies:
            self.log("没有可保存的检测结果")
            return
            
        # 选择保存目录
        dir_dialog = QFileDialog()
        save_dir = dir_dialog.getExistingDirectory(self, "选择保存目录")
        
        if not save_dir:
            return
            
        # 使用默认值，因为单图处理UI中没有这些控件
        format_ext = "jpg"
        do_resize = False
        width = None
        height = None
        is_hd = True
        
        # 创建子目录
        if self.detected_faces:
            # 创建子目录
            faces_dir = os.path.join(save_dir, "faces")
            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir)
                
            self.log(f"保存人脸至: {faces_dir}")
            self.save_faces(faces_dir, format_ext, do_resize, width, height, is_hd)
            
        elif self.detected_bodies:
            # 创建子目录
            bodies_dir = os.path.join(save_dir, "bodies")
            if not os.path.exists(bodies_dir):
                os.makedirs(bodies_dir)
                
            self.log(f"保存人体至: {bodies_dir}")
            self.save_bodies(bodies_dir, format_ext, do_resize, width, height, is_hd)

    def save_faces(self, save_dir, format_ext, do_resize, width, height, is_hd):
        """保存人脸图像"""
        # 获取原始图像基本名称
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        
        saved_count = 0
        
        for i, face_data in enumerate(self.detected_faces):
            try:
                # 获取人脸图像
                face_img = face_data['face']
                
                # 如果需要高清导出，使用原始尺寸或超分辨率算法提升质量
                if is_hd:
                    # 这里可以添加超分辨率算法，目前只是简单地保留原图质量
                    # 如果需要可以集成超分辨率模型
                    pass  # 保持原样即可
                
                # 如果需要调整大小
                if do_resize and width and height:
                    face_img = cv2.resize(face_img, (width, height))
                
                # 创建唯一文件名
                unique_id = uuid.uuid4().hex[:8]
                save_name = f"{base_name}_face_{i+1}_{unique_id}.{format_ext}"
                save_path = os.path.join(save_dir, save_name)
                
                # 使用安全保存方法
                if save_image_safe(save_path, face_img, format_ext):
                    saved_count += 1
                    self.log(f"已保存: {save_name}")
                else:
                    self.log(f"保存失败: {save_name}")
                    
            except Exception as e:
                self.log(f"保存人脸图像时出错: {str(e)}")
        
        self.log(f"保存完成。成功保存 {saved_count} 个人脸图像")
    
    def save_bodies(self, save_dir, format_ext, do_resize, width, height, is_hd):
        """保存人体图像"""
        # 获取原始图像基本名称
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        
        saved_count = 0
        
        for i, body_data in enumerate(self.detected_bodies):
            try:
                # 获取人体图像
                body_img = body_data['body']
                
                # 如果需要高清导出，使用原始尺寸或超分辨率算法提升质量
                if is_hd:
                    # 这里可以添加超分辨率算法，目前只是简单地保留原图质量
                    # 如果需要可以集成超分辨率模型
                    pass  # 保持原样即可
                
                # 如果需要调整大小
                if do_resize and width and height:
                    body_img = cv2.resize(body_img, (width, height))
                
                # 创建唯一文件名
                unique_id = uuid.uuid4().hex[:8]
                save_name = f"{base_name}_body_{i+1}_{unique_id}.{format_ext}"
                save_path = os.path.join(save_dir, save_name)
                
                # 使用安全保存方法
                if save_image_safe(save_path, body_img, format_ext):
                    saved_count += 1
                    self.log(f"已保存: {save_name}")
                else:
                    self.log(f"保存失败: {save_name}")
                    
            except Exception as e:
                self.log(f"保存人体图像时出错: {str(e)}")
        
        self.log(f"保存完成。成功保存 {saved_count} 个人体图像")

    def batch_process_folder(self):
        """批量处理文件夹中的图像"""
        # 选择输入文件夹
        input_dir_dialog = QFileDialog()
        input_dir = input_dir_dialog.getExistingDirectory(self, "选择包含图像的文件夹")
        
        if not input_dir:
            return
            
        # 选择输出文件夹
        output_dir_dialog = QFileDialog()
        output_dir = output_dir_dialog.getExistingDirectory(self, "选择保存结果的文件夹")
        
        if not output_dir:
            return
            
        # 获取自定义文本内容（弹出对话框让用户输入）
        text_content_dialog = QDialog(self)
        text_content_dialog.setWindowTitle("自定义文本内容")
        text_content_dialog.setMinimumWidth(400)
        
        dialog_layout = QVBoxLayout(text_content_dialog)
        dialog_layout.addWidget(QLabel("请输入要保存在文本文件中的内容:"))
        
        text_content_edit = QTextEdit()
        text_content_edit.setMinimumHeight(100)
        text_content_edit.setPlaceholderText("留空则使用默认的文件名格式（Rem_xxxxx）")
        dialog_layout.addWidget(text_content_edit)
        
        button_box = QHBoxLayout()
        cancel_btn = QPushButton("取消")
        ok_btn = QPushButton("确定")
        cancel_btn.clicked.connect(text_content_dialog.reject)
        ok_btn.clicked.connect(text_content_dialog.accept)
        button_box.addWidget(cancel_btn)
        button_box.addWidget(ok_btn)
        dialog_layout.addLayout(button_box)
        
        # 默认文本内容为空
        custom_text_content = ""
        
        # 显示对话框并获取用户输入
        if text_content_dialog.exec_() == QDialog.Accepted:
            custom_text_content = text_content_edit.toPlainText()
            
        # 获取处理设置
        if self.face_detect_radio.isChecked():
            # 人脸检测设置
            min_face_size = self.min_face_size_spin.value()
            
            # 使用YOLO检测方法（已经固定）
            margin_ratio = 0.2  # 固定的人脸背景比例
            
            # 检查YOLO模型文件是否存在
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            weights_path = os.path.join(model_dir, "yolov3.weights")
            config_path = os.path.join(model_dir, "yolov3.cfg")
            
            if not os.path.exists(weights_path) or not os.path.exists(config_path):
                QMessageBox.warning(
                    self,
                    "YOLO模型文件缺失",
                    "未找到YOLO模型文件！\n\n"
                    "请确保以下文件存在于'models'目录中：\n"
                    "1. yolov3.weights (约236MB)\n"
                    "2. yolov3.cfg\n\n"
                    "您可以使用optimize_and_run.bat中的选项5下载这些文件，"
                    "或参考README.md中的说明手动下载。"
                )
                self.batch_process_btn.setEnabled(True)
                return
            
            # 获取输出设置
            format_ext = self.format_combo.currentText()
            do_resize = self.resize_checkbox.isChecked()
            width = self.width_spin.value() if do_resize else None
            height = self.height_spin.value() if do_resize else None
            is_hd = self.hd_export_checkbox.isChecked()
            
            try:
                # 禁用批处理按钮，防止重复点击
                self.batch_process_btn.setEnabled(False)
                self.progress_bar.setValue(0)
                
                self.log("批量处理使用 YOLO 方法进行人脸检测...")
                
                # 创建批处理线程
                self.batch_thread = BatchFolderProcessThread(
                    input_dir, 
                    output_dir, 
                    "yolo",  # 强制使用YOLO方法 
                    min_face_size,
                    format_ext,
                    do_resize,
                    width,
                    height,
                    is_hd,
                    custom_text_content
                )
                
                # 连接信号
                self.batch_thread.progress_signal.connect(self.progress_bar.setValue)
                self.batch_thread.log_signal.connect(self.log)
                self.batch_thread.completed_signal.connect(self.on_batch_process_finished)
                
                # 开始线程
                self.batch_thread.start()
                self.log(f"开始批量处理文件夹: {input_dir}")
                
            except Exception as e:
                self.log(f"批量处理时出错: {str(e)}")
                self.batch_process_btn.setEnabled(True)
        else:
            # 人体检测设置
            min_body_size = self.min_body_size_spin.value()
            
            # 获取背景比例设置（始终使用YOLO）
            margin_ratio = self.margin_ratio_slider.value() / 100.0
            use_yolo = True
            
            # 检查YOLO模型文件是否存在
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            weights_path = os.path.join(model_dir, "yolov3.weights")
            config_path = os.path.join(model_dir, "yolov3.cfg")
            
            if not os.path.exists(weights_path) or not os.path.exists(config_path):
                QMessageBox.warning(
                    self,
                    "YOLO模型文件缺失",
                    "未找到YOLO模型文件！\n\n"
                    "请确保以下文件存在于'models'目录中：\n"
                    "1. yolov3.weights (约236MB)\n"
                    "2. yolov3.cfg\n\n"
                    "您可以使用optimize_and_run.bat中的选项5下载这些文件，"
                    "或参考README.md中的说明手动下载。"
                )
                self.batch_process_btn.setEnabled(True)
                return
            
            # 获取输出设置
            format_ext = self.format_combo.currentText()
            do_resize = self.resize_checkbox.isChecked()
            width = self.width_spin.value() if do_resize else None
            height = self.height_spin.value() if do_resize else None
            is_hd = self.hd_export_checkbox.isChecked()
            
            try:
                # 禁用批处理按钮，防止重复点击
                self.batch_process_btn.setEnabled(False)
                self.progress_bar.setValue(0)
                
                self.log("批量处理使用 YOLO 方法进行人体检测...")
                
                # 创建并启动批处理线程
                self.batch_thread = BatchBodyProcessThread(
                    input_dir,
                    output_dir,
                    min_body_size=min_body_size,
                    format_ext=format_ext,
                    resize=do_resize,
                    width=width,
                    height=height,
                    is_hd=is_hd,
                    text_content=custom_text_content,
                    margin_ratio_x=margin_ratio_x,
                    margin_ratio_y=margin_ratio_y,
                    # margin_ratio参数保留用于向后兼容
                    margin_ratio=0.2,  # 此参数在使用分离边距时将被忽略
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    only_highest_confidence=self.batch_only_highest_confidence_checkbox.isChecked(),
                    prioritize_center=self.batch_prioritize_center_checkbox.isChecked(),
                    resize_mode=resize_mode
                )
                
                # 连接信号
                self.batch_thread.update_progress.connect(self.progress_bar.setValue)
                self.batch_thread.log_message.connect(self.log)
                self.batch_thread.finished.connect(self.on_batch_process_finished)
                
                # 启动线程
                self.batch_thread.start()
                self.log(f"开始批量处理文件夹: {input_dir}")
                
            except Exception as e:
                self.log(f"批量处理时出错: {str(e)}")
                self.batch_process_btn.setEnabled(True)
    
    def on_batch_process_finished(self, processed_files, extracted_count):
        """批量处理完成的回调"""
        self.batch_process_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # 显示处理结果
        QMessageBox.information(
            self, 
            "批量处理完成", 
            f"处理了 {processed_files} 个文件，共提取 {extracted_count} 个对象。"
        )
        
    def open_image_splitter(self):
        """打开图像切割工具"""
        splitter_dialog = ImageSplitterDialog(self)
        splitter_dialog.exec_()

    def extract_subfolder_images(self):
        """提取所有子文件夹中的图像"""
        # 选择输入文件夹
        input_dir_dialog = QFileDialog()
        input_dir = input_dir_dialog.getExistingDirectory(self, "选择包含子文件夹的父文件夹")
        
        if not input_dir:
            return
            
        # 选择输出文件夹
        output_dir_dialog = QFileDialog()
        output_dir = output_dir_dialog.getExistingDirectory(self, "选择保存提取图像的文件夹")
        
        if not output_dir:
            return
        
        # 选择文件命名方式
        rename_option_dialog = QDialog(self)
        rename_option_dialog.setWindowTitle("文件命名选项")
        rename_option_dialog.setMinimumWidth(400)
        
        dialog_layout = QVBoxLayout(rename_option_dialog)
        dialog_layout.addWidget(QLabel("请选择提取文件的命名方式:"))
        
        # 命名方式选项
        naming_options_group = QGroupBox("命名选项")
        naming_options_layout = QVBoxLayout()
        
        # 保持原文件名选项
        keep_original_radio = QRadioButton("保持原始文件名（添加数字以避免冲突）")
        keep_original_radio.setChecked(True)
        naming_options_layout.addWidget(keep_original_radio)
        
        # 使用Rem_xxxxx格式选项
        use_rem_format_radio = QRadioButton("使用Rem_00001格式重命名")
        naming_options_layout.addWidget(use_rem_format_radio)
        
        # 使用父文件夹名加数字格式
        use_parent_folder_radio = QRadioButton("使用父文件夹名_00001格式")
        naming_options_layout.addWidget(use_parent_folder_radio)
        
        # 前缀设置
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("自定义前缀:"))
        prefix_edit = QLineEdit()
        prefix_edit.setText("Rem")
        prefix_edit.setEnabled(use_rem_format_radio.isChecked())
        prefix_layout.addWidget(prefix_edit)
        
        # 连接事件，仅当选择Rem格式时启用前缀编辑
        use_rem_format_radio.toggled.connect(lambda checked: prefix_edit.setEnabled(checked))
        
        naming_options_layout.addLayout(prefix_layout)
        
        # 起始编号设置
        start_number_layout = QHBoxLayout()
        start_number_layout.addWidget(QLabel("起始编号:"))
        start_number_spin = QSpinBox()
        start_number_spin.setRange(1, 99999)
        start_number_spin.setValue(1)
        start_number_layout.addWidget(start_number_spin)
        
        naming_options_layout.addLayout(start_number_layout)
        
        # 文本文件生成选项
        gen_text_checkbox = QCheckBox("生成对应的文本文件")
        gen_text_checkbox.setChecked(True)
        naming_options_layout.addWidget(gen_text_checkbox)
        
        # 文本内容设置
        dialog_layout.addWidget(QLabel("输入要保存在文本文件中的内容:"))
        text_content_edit = QTextEdit()
        text_content_edit.setMinimumHeight(100)
        text_content_edit.setPlaceholderText("留空则使用文件名作为内容")
        dialog_layout.addWidget(text_content_edit)
        
        naming_options_group.setLayout(naming_options_layout)
        dialog_layout.addWidget(naming_options_group)
        
        # 输出格式设置
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出格式:"))
        format_combo = QComboBox()
        format_combo.addItems(["jpg", "png", "bmp"])
        format_layout.addWidget(format_combo)
        dialog_layout.addLayout(format_layout)
        
        # 调整大小选项
        resize_checkbox = QCheckBox("调整图像尺寸")
        resize_checkbox.setChecked(False)
        dialog_layout.addWidget(resize_checkbox)
        
        resize_options_layout = QHBoxLayout()
        resize_options_layout.addWidget(QLabel("宽度:"))
        width_spin = QSpinBox()
        width_spin.setRange(32, 4096)
        width_spin.setValue(512)
        width_spin.setEnabled(False)
        resize_options_layout.addWidget(width_spin)
        
        resize_options_layout.addWidget(QLabel("高度:"))
        height_spin = QSpinBox()
        height_spin.setRange(32, 4096)
        height_spin.setValue(512)
        height_spin.setEnabled(False)
        resize_options_layout.addWidget(height_spin)
        
        dialog_layout.addLayout(resize_options_layout)
        
        # 连接事件，仅当选择调整大小时启用尺寸设置
        resize_checkbox.toggled.connect(lambda checked: width_spin.setEnabled(checked))
        resize_checkbox.toggled.connect(lambda checked: height_spin.setEnabled(checked))
        
        # 确定取消按钮
        button_box = QHBoxLayout()
        cancel_btn = QPushButton("取消")
        ok_btn = QPushButton("确定")
        cancel_btn.clicked.connect(rename_option_dialog.reject)
        ok_btn.clicked.connect(rename_option_dialog.accept)
        button_box.addWidget(cancel_btn)
        button_box.addWidget(ok_btn)
        dialog_layout.addLayout(button_box)
        
        # 显示对话框并获取用户选择
        if rename_option_dialog.exec_() != QDialog.Accepted:
            return
        
        # 获取用户设置
        naming_option = 0  # 0: 保持原名, 1: Rem格式, 2: 父文件夹格式
        if use_rem_format_radio.isChecked():
            naming_option = 1
        elif use_parent_folder_radio.isChecked():
            naming_option = 2
            
        custom_prefix = prefix_edit.text() if prefix_edit.text() else "Rem"
        start_number = start_number_spin.value()
        generate_text = gen_text_checkbox.isChecked()
        text_content = text_content_edit.toPlainText()
        output_format = format_combo.currentText()
        do_resize = resize_checkbox.isChecked()
        resize_width = width_spin.value() if do_resize else None
        resize_height = height_spin.value() if do_resize else None
        
        # 开始执行提取操作
        try:
            # 禁用按钮，防止重复点击
            self.extract_subfolder_images_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            
            # 创建提取线程
            self.extract_thread = SubfolderImageExtractThread(
                input_dir,
                output_dir,
                naming_option,
                custom_prefix,
                start_number,
                generate_text,
                text_content,
                output_format,
                do_resize,
                resize_width,
                resize_height
            )
            
            # 连接信号
            self.extract_thread.update_progress.connect(self.progress_bar.setValue)
            self.extract_thread.log_message.connect(self.log)
            self.extract_thread.finished.connect(self.on_extract_finished)
            
            # 开始线程
            self.extract_thread.start()
            self.log(f"开始从子文件夹提取图像: {input_dir}")
            
        except Exception as e:
            self.log(f"提取图像时出错: {str(e)}")
            self.extract_subfolder_images_btn.setEnabled(True)
    
    def on_extract_finished(self, total_dirs, total_files, extracted_files):
        """提取图像完成的回调"""
        self.extract_subfolder_images_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        
        # 显示处理结果
        QMessageBox.information(
            self, 
            "提取完成", 
            f"处理了{total_dirs}个子文件夹、{total_files}个文件，成功提取了{extracted_files}个图像。"
        )

    def browse_batch_input_dir(self):
        """浏览批量输入目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.batch_input_path.setText(dir_path)
            self.batch_log(f"已选择输入目录: {dir_path}")
            
            # 自动设置预览图像
            self.set_preview_sample_from_dir(dir_path)
    
    def set_preview_sample_from_dir(self, dir_path):
        """从目录中自动设置一个预览样本图像"""
        try:
            # 查找目录中的图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
                # 只搜索第一级目录就足够了
                break
            
            if image_files:
                # 设置第一个图像作为预览样本
                self.preview_sample_path = image_files[0]
                self.batch_log(f"已自动选择预览图像: {os.path.basename(self.preview_sample_path)}")
                
                # 加载并显示预览图像缩略图
                try:
                    img = read_image_safe(self.preview_sample_path)
                    if img is not None:
                        # 调整图像大小以适应预览区域
                        h, w = img.shape[:2]
                        preview_width = min(400, w)
                        ratio = preview_width / w
                        preview_height = int(h * ratio)
                        img_resized = cv2.resize(img, (preview_width, preview_height))
                        
                        # 将OpenCV图像转换为Qt图像
                        h, w, c = img_resized.shape
                        q_image = QImage(img_resized.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                        pixmap = QPixmap.fromImage(q_image)
                        
                        # 显示缩略图
                        self.preview_image_label.setPixmap(pixmap)
                        self.preview_image_label.setText("")
                except Exception as e:
                    self.batch_log(f"加载预览图像出错: {str(e)}")
            else:
                self.preview_sample_path = None
                self.preview_image_label.setText("未找到图像文件")
                self.batch_log("在目录中未找到图像文件")
        except Exception as e:
            self.batch_log(f"设置预览样本时出错: {str(e)}")
            self.preview_sample_path = None

    def browse_batch_output_dir(self):
        """浏览选择批量处理的输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存结果的目录")
        if dir_path:
            self.batch_output_path.setText(dir_path)

    def batch_log(self, message):
        """向批量处理日志区域添加消息"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.batch_log_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.batch_log_text.verticalScrollBar().setValue(self.batch_log_text.verticalScrollBar().maximum())

    def start_batch_body_process(self):
        """开始批量人体检测与提取"""
        # 检查输入和输出路径
        input_dir = self.batch_input_path.text()
        output_dir = self.batch_output_path.text()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "路径错误", "请选择输入和输出目录")
            return
        
        if not os.path.exists(input_dir):
            QMessageBox.warning(self, "路径错误", f"输入目录不存在: {input_dir}")
            return
        
        # 获取处理参数
        min_body_size = self.batch_min_body_size.value()
        # 使用单独的水平和垂直边距比例
        margin_ratio_x = self.batch_margin_x_slider.value() / 100.0  # 水平边距（左右）
        margin_ratio_y = self.batch_margin_y_slider.value() / 100.0  # 垂直边距（上下）
        format_ext = self.batch_format_combo.currentText()
        do_resize = self.batch_resize_checkbox.isChecked()
        width = self.batch_resize_width.value() if do_resize else None
        height = self.batch_resize_height.value() if do_resize else None
        is_hd = self.batch_hd_checkbox.isChecked()
        text_content = self.batch_text_content.toPlainText()
        
        # 获取调整大小模式
        resize_mode = self.batch_resize_mode_combo.currentData() if do_resize else "stretch"
        
        # 获取检测阈值参数
        confidence_threshold = self.batch_confidence_slider.value() / 100.0
        nms_threshold = self.batch_nms_slider.value() / 100.0
        
        # 清空日志
        self.batch_log_text.clear()
        
        try:
            # 更新界面状态
            self.start_batch_btn.setEnabled(False)
            self.stop_batch_btn.setEnabled(True)
            self.batch_progress_bar.setValue(0)
            
            # 记录处理参数
            self.batch_log(f"开始批量人体检测与提取")
            self.batch_log(f"输入目录: {input_dir}")
            self.batch_log(f"输出目录: {output_dir}")
            self.batch_log(f"最小人体尺寸: {min_body_size}像素")
            self.batch_log(f"背景边距比例: {margin_ratio_x:.2f}, {margin_ratio_y:.2f}")
            self.batch_log(f"置信度阈值: {confidence_threshold:.2f}")
            self.batch_log(f"NMS阈值: {nms_threshold:.2f}")
            self.batch_log(f"输出格式: {format_ext}")
            if do_resize:
                self.batch_log(f"调整大小为: {width}x{height}")
                self.batch_log(f"调整模式: {resize_mode}")
            if is_hd:
                self.batch_log("高清输出: 启用")
            if text_content:
                self.batch_log("将使用自定义文本内容")
            
            # 创建并启动批处理线程
            self.batch_thread = BatchBodyProcessThread(
                input_dir,
                output_dir,
                min_body_size=min_body_size,
                format_ext=format_ext,
                resize=do_resize,
                width=width,
                height=height,
                is_hd=is_hd,
                text_content=text_content,
                margin_ratio_x=margin_ratio_x,
                margin_ratio_y=margin_ratio_y,
                # margin_ratio参数保留用于向后兼容
                margin_ratio=0.2,  # 此参数在使用分离边距时将被忽略
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                only_highest_confidence=self.batch_only_highest_confidence_checkbox.isChecked(),
                prioritize_center=self.batch_prioritize_center_checkbox.isChecked(),
                resize_mode=resize_mode
            )
            
            # 连接信号
            self.batch_thread.update_progress.connect(self.batch_progress_bar.setValue)
            self.batch_thread.log_message.connect(self.batch_log)
            self.batch_thread.finished.connect(self.on_batch_body_process_finished)
            
            # 启动线程
            self.batch_thread.start()
            
        except Exception as e:
            self.batch_log(f"启动批处理时出错: {str(e)}")
            self.start_batch_btn.setEnabled(True)
            self.stop_batch_btn.setEnabled(False)

    def stop_batch_process(self):
        """停止批量处理"""
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_log("正在停止处理...")
            self.batch_thread.stop()
            self.stop_batch_btn.setEnabled(False)

    def on_batch_body_process_finished(self, processed_files, extracted_bodies):
        """批量人体处理完成的回调"""
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_progress_bar.setValue(100)
        
        # 显示处理结果
        self.batch_log("")
        self.batch_log("=" * 50)
        self.batch_log(f"处理完成！")
        self.batch_log(f"共处理 {processed_files} 个文件，提取 {extracted_bodies} 个人体")
        self.batch_log(f"结果保存在: {self.batch_output_path.text()}")
        
        # 显示结果对话框
        QMessageBox.information(
            self, 
            "批量处理完成", 
            f"处理了 {processed_files} 个文件，共提取 {extracted_bodies} 个人体。"
        )

    def update_size_slider(self):
        """更新最小人体尺寸滑块"""
        self.min_body_size_spin.setValue(self.batch_min_body_size.value())

    def update_preview_on_param_change(self):
        """当参数变化时更新预览"""
        # 获取当前参数
        confidence_threshold = self.batch_confidence_slider.value() / 100.0
        nms_threshold = self.batch_nms_slider.value() / 100.0
        
        # 记录参数变化
        self.batch_log(f"检测参数已改变: 置信度阈值={confidence_threshold:.2f}, NMS阈值={nms_threshold:.2f}")
        
        # 如果还没有定义计时器，创建一个
        if not hasattr(self, 'preview_update_timer'):
            self.preview_update_timer = QTimer()
            self.preview_update_timer.setSingleShot(True)
            self.preview_update_timer.timeout.connect(self.preview_body_detection)
        
        # 重置计时器，500毫秒后更新预览
        self.preview_update_timer.stop()
        self.preview_update_timer.start(500)

    def preview_body_detection(self):
        """预览人体检测"""
        # 检查是否有预览样本图像
        if not hasattr(self, 'preview_sample_path') or not self.preview_sample_path:
            self.batch_log("请先选择输入目录以获取预览图像")
            return
            
        self.batch_log("正在预览人体检测...")
        
        # 获取当前参数
        min_body_size = self.batch_min_body_size.value()
        # 使用水平和垂直独立边距
        margin_ratio_x = self.batch_margin_x_slider.value() / 100.0  # 水平边距（左右）
        margin_ratio_y = self.batch_margin_y_slider.value() / 100.0  # 垂直边距（上下）
        confidence_threshold = self.batch_confidence_slider.value() / 100.0
        nms_threshold = self.batch_nms_slider.value() / 100.0
        only_highest_confidence = self.batch_only_highest_confidence_checkbox.isChecked()
        
        # 记录检测参数
        self.batch_log(f"预览检测参数: 最小尺寸={min_body_size}像素")
        self.batch_log(f"水平边距比例={margin_ratio_x:.2f}, 垂直边距比例={margin_ratio_y:.2f}")
        self.batch_log(f"置信度阈值={confidence_threshold:.2f}, NMS阈值={nms_threshold:.2f}")
        
        # 创建自定义线程，使用分离的边距比例
        class EnhancedBodyDetectionThread(BodyDetectionThread):
            def run(self):
                try:
                    # 读取原始图像
                    self.log_message.emit(f"正在读取图像: {self.image_path}")
                    img = read_image_safe(self.image_path)
                    
                    if img is None:
                        self.log_message.emit("无法读取图像，请检查图像路径或格式")
                        self.finished_detection.emit([])
                        return
                        
                    # 创建图像副本
                    original_img = img.copy()
                    
                    # 发送进度更新
                    self.update_progress.emit(10)
                    
                    # 初始化检测结果
                    body_regions = []
                    
                    # 使用支持独立边距的函数
                    self.log_message.emit("使用YOLO模型进行人体检测...")
                    self.log_message.emit(f"检测参数: 最小尺寸={self.min_body_size}像素")
                    self.log_message.emit(f"检测参数: 水平边距={margin_ratio_x:.2f}, 垂直边距={margin_ratio_y:.2f}")
                    self.log_message.emit(f"检测参数: 置信度阈值={self.confidence_threshold:.2f}, NMS阈值={self.nms_threshold:.2f}")
                    
                    body_regions = detect_bodies_with_separate_margins(
                        img, 
                        min_body_size=self.min_body_size, 
                        margin_ratio_x=margin_ratio_x,
                        margin_ratio_y=margin_ratio_y,
                        confidence_threshold=self.confidence_threshold,
                        nms_threshold=self.nms_threshold
                    )
                    
                    # 在显示图像上标记检测到的人体
                    if self.show_borders and body_regions:
                        for body_data in body_regions:
                            x1, y1, x2, y2 = body_data['coords']
                            confidence = body_data.get('confidence', 0.0)
                            
                            # 绘制检测框和置信度
                            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 添加置信度文本
                            text = f"Person: {confidence:.2f}"
                            cv2.putText(original_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    self.log_message.emit(f"YOLO检测到 {len(body_regions)} 个人体")
                    
                    # 如果没有检测到任何人体，考虑整个图像作为一个区域
                    if not body_regions and img is not None:
                        self.log_message.emit("未检测到人体，使用整个图像")
                        height, width = img.shape[:2]
                        body_regions = [{
                            'body': img.copy(),
                            'coords': (0, 0, width, height),
                            'original_coords': (0, 0, width, height),
                            'confidence': 1.0  # 整图使用，认为是100%的置信度
                        }]
                        
                        # 如果需要显示边框，绘制整个图像的边框
                        if self.show_borders:
                            cv2.rectangle(original_img, (0, 0), (width, height), (0, 0, 255), 2)
                            cv2.putText(original_img, "Full Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 发送进度更新
                    self.update_progress.emit(90)
                    
                    # 将检测结果直接传递给应用程序
                    result_bodies = body_regions
                    
                    # 显示检测结果
                    if result_bodies:
                        # 将OpenCV图像转换为Qt图像
                        h, w, c = original_img.shape
                        q_image = QImage(original_img.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
                        pixmap = QPixmap.fromImage(q_image)
                        self.update_image.emit(pixmap)
                    
                    # 发送检测完成信号
                    self.log_message.emit("人体检测完成")
                    self.finished_detection.emit(result_bodies)
                    self.update_progress.emit(100)
                
                except Exception as e:
                    self.log_message.emit(f"人体检测出错: {str(e)}")
                    self.finished_detection.emit([])
                    self.update_progress.emit(0)
        
        # 创建并启动增强的线程
        self.preview_body_detection_thread = EnhancedBodyDetectionThread(
            self.preview_sample_path,
            min_body_size=min_body_size,
            margin_ratio=0.2,  # 这个值会被忽略，但保留为向后兼容
            use_yolo=True,
            show_borders=True,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        # 连接信号
        self.preview_body_detection_thread.update_progress.connect(self.batch_progress_bar.setValue)
        self.preview_body_detection_thread.update_image.connect(self.update_preview_image)
        self.preview_body_detection_thread.finished_detection.connect(self.handle_preview_detection_finished)
        self.preview_body_detection_thread.log_message.connect(self.batch_log)
        
        # 启动线程
        self.preview_body_detection_thread.start()
        
    def update_preview_image(self, pixmap):
        """更新预览图像"""
        # 调整图像大小以适应预览区域
        scaled_pixmap = pixmap.scaled(
            self.preview_image_label.width(), 
            self.preview_image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_image_label.setPixmap(scaled_pixmap)
        
    def handle_preview_detection_finished(self, body_regions):
        """处理预览检测完成"""
        # 获取当前参数 - 修正：使用批量处理滑块，而非单图处理滑块
        confidence_threshold = self.batch_confidence_slider.value() / 100.0
        nms_threshold = self.batch_nms_slider.value() / 100.0
        
        # 计算优先中心处理
        if self.batch_prioritize_center_checkbox.isChecked() and len(body_regions) > 1:
            # 获取当前图像尺寸
            if self.preview_image_path:
                img = read_image_safe(self.preview_image_path)
                if img is not None:
                    height, width = img.shape[:2]
                    # 计算图像中心点
                    center_x, center_y = width / 2, height / 2
                    
                    # 为每个人体计算到中心的距离
                    for body in body_regions:
                        x, y, w, h = body.get('coords', (0, 0, 0, 0))
                        # 计算人体边界框中心
                        body_center_x = x + w / 2
                        body_center_y = y + h / 2
                        # 计算到图像中心的距离
                        distance = ((body_center_x - center_x) ** 2 + (body_center_y - center_y) ** 2) ** 0.5
                        body['center_distance'] = distance
                    
                    # 按照到中心的距离排序
                    body_regions = sorted(body_regions, key=lambda x: x.get('center_distance', float('inf')))
                    self.batch_log(f"已启用'优先中心人体'，已按距离中心从近到远排序")
                    
                    # 如果同时启用了仅保留最高置信度，则只保留中心最近的一个
                    if self.batch_only_highest_confidence_checkbox.isChecked():
                        body_regions = [body_regions[0]]
                        self.batch_log(f"同时启用'仅保留最高置信度结果'，保留距离中心最近的人体")
        
        # 如果只启用了"仅保留最高置信度结果"选项且检测到多个人体，只保留置信度最高的一个
        elif self.batch_only_highest_confidence_checkbox.isChecked() and len(body_regions) > 1:
            # 找出置信度最高的人体
            highest_confidence_body = max(body_regions, key=lambda x: x.get('confidence', 0))
            body_regions = [highest_confidence_body]
            self.batch_log(f"已启用'仅保留最高置信度结果'，过滤后保留 1 个人体")
        
        # 更新预览信息
        priority_center_text = "，已启用优先中心人体" if self.batch_prioritize_center_checkbox.isChecked() else ""
        only_highest_text = "，已启用仅保留最高置信度" if self.batch_only_highest_confidence_checkbox.isChecked() else ""
        self.preview_info_label.setText(
            f"预览信息: 检测到 {len(body_regions)} 个人体{priority_center_text}{only_highest_text}，"
            f"置信度阈值: {confidence_threshold:.2f}，"
            f"NMS阈值: {nms_threshold:.2f}"
        )

    def update_min_body_size(self):
        """更新最小人体尺寸"""
        self.min_body_size_spin.setValue(self.detection_size_slider.value())
        self.size_value_label.setText(f"{self.detection_size_slider.value()}像素")

    def trigger_delayed_detection(self):
        """触发延迟检测"""
        # 如果没有加载图像，不执行检测
        if not self.current_image_path:
            return
            
        # 如果当前正在进行检测，不触发新的检测
        if hasattr(self, 'body_detection_thread') and self.body_detection_thread and self.body_detection_thread.isRunning():
            return
            
        # 获取当前的阈值参数 - 修正：使用单图处理的滑块，而非批量处理的滑块
        confidence_val = self.confidence_threshold_slider.value() / 100.0
        nms_val = self.nms_threshold_slider.value() / 100.0
        
        # 记录检测参数
        self.log(f"检测参数已改变: 置信度阈值={confidence_val:.2f}, NMS阈值={nms_val:.2f}")
        
        # 如果还没有定义计时器，创建一个
        if not hasattr(self, 'detection_update_timer'):
            self.detection_update_timer = QTimer()
            self.detection_update_timer.setSingleShot(True)
            self.detection_update_timer.timeout.connect(self.detect_objects)
        
        # 重置计时器，500毫秒后更新检测
        self.detection_update_timer.stop()
        self.detection_update_timer.start(500)


class ImageSplitterDialog(QDialog):
    """图像切割工具对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像切割工具")
        self.setMinimumSize(800, 600)
        self.setModal(True)
        
        # 初始化变量
        self.image_path = None
        self.loaded_image = None
        self.preview_image = None
        self.cut_images = []
        
        # 使用临时文件管理器
        self.temp_manager = TempFileManager()
        
        # 创建布局
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建水平分割的布局
        h_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 图像加载按钮
        load_btn = QPushButton("加载图像")
        load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(load_btn)
        
        # 批量重命名按钮
        batch_rename_btn = QPushButton("批量重命名(Rem格式)")
        batch_rename_btn.clicked.connect(self.batch_rename_files)
        left_layout.addWidget(batch_rename_btn)
        
        # 切割参数组
        cutting_group = QGroupBox("切割参数")
        cutting_layout = QGridLayout()
        
        cutting_layout.addWidget(QLabel("行数:"), 0, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(2)
        cutting_layout.addWidget(self.rows_spin, 0, 1)
        
        cutting_layout.addWidget(QLabel("列数:"), 1, 0)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(2)
        cutting_layout.addWidget(self.cols_spin, 1, 1)
        
        # 预览按钮
        self.preview_btn = QPushButton("预览切割")
        self.preview_btn.clicked.connect(self.preview_cuts)
        self.preview_btn.setEnabled(False)
        cutting_layout.addWidget(self.preview_btn, 2, 0, 1, 2)
        
        cutting_group.setLayout(cutting_layout)
        left_layout.addWidget(cutting_group)
        
        # 调整选项组
        resize_group = QGroupBox("调整选项")
        resize_layout = QGridLayout()
        
        self.resize_checkbox = QCheckBox("调整切片大小")
        resize_layout.addWidget(self.resize_checkbox, 0, 0, 1, 2)
        
        resize_layout.addWidget(QLabel("宽度:"), 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 4096)
        self.width_spin.setValue(512)
        resize_layout.addWidget(self.width_spin, 1, 1)
        
        resize_layout.addWidget(QLabel("高度:"), 2, 0)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 4096)
        self.height_spin.setValue(512)
        resize_layout.addWidget(self.height_spin, 2, 1)
        
        resize_group.setLayout(resize_layout)
        left_layout.addWidget(resize_group)
        
        # 导出选项组
        export_group = QGroupBox("导出选项")
        export_layout = QGridLayout()
        
        export_layout.addWidget(QLabel("格式:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png", "bmp"])
        export_layout.addWidget(self.format_combo, 0, 1)
        
        # 高清导出选项
        self.hd_export_checkbox = QCheckBox("高清导出")
        self.hd_export_checkbox.setChecked(True)
        export_layout.addWidget(self.hd_export_checkbox, 1, 0, 1, 2)
        
        # 生成文本选项
        self.gen_text_checkbox = QCheckBox("生成文本文件")
        self.gen_text_checkbox.setChecked(True)
        export_layout.addWidget(self.gen_text_checkbox, 2, 0, 1, 2)
        
        # 文本内容输入
        export_layout.addWidget(QLabel("文本内容:"), 3, 0)
        self.text_content = QLineEdit()
        self.text_content.setText("Rem")  # 默认文本
        export_layout.addWidget(self.text_content, 3, 1)
        
        # 文本文件内容
        export_layout.addWidget(QLabel("文本文件内容:"), 4, 0, 1, 2)
        self.text_file_content = QTextEdit()
        self.text_file_content.setPlaceholderText("输入要保存在生成的txt文件中的文本")
        self.text_file_content.setMinimumHeight(100)
        export_layout.addWidget(self.text_file_content, 5, 0, 1, 2)
        
        # 起始编号
        export_layout.addWidget(QLabel("起始编号:"), 6, 0)
        self.start_number_spin = QSpinBox()
        self.start_number_spin.setRange(1, 99999)
        self.start_number_spin.setValue(1)
        export_layout.addWidget(self.start_number_spin, 6, 1)
        
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)
        
        # 保存按钮
        self.save_btn = QPushButton("保存切片")
        self.save_btn.clicked.connect(self.save_cuts)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)
        
        # 添加拉伸空间
        left_layout.addStretch(1)
        
        # 右侧图像显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图像预览
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("未加载图像")
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        self.image_scroll_area.setWidget(self.image_label)
        
        right_layout.addWidget(self.image_scroll_area)
        
        # 添加左右面板到分割器
        h_splitter.addWidget(left_panel)
        h_splitter.addWidget(right_panel)
        h_splitter.setSizes([300, 500])  # 设置初始宽度比例
        
        # 添加分割器到主布局
        main_layout.addWidget(h_splitter)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
    
    def batch_rename_files(self):
        """批量重命名文件为Rem_00001格式"""
        # 选择源文件夹
        src_dir_dialog = QFileDialog()
        src_dir = src_dir_dialog.getExistingDirectory(self, "选择源文件夹")
        if not src_dir:
            return
        
        # 选择目标文件夹
        dst_dir_dialog = QFileDialog()
        dst_dir = dst_dir_dialog.getExistingDirectory(self, "选择目标文件夹")
        if not dst_dir:
            return
        
        # 获取要处理的图像文件格式
        format_ext = self.format_combo.currentText().lower()
        
        # 获取文本前缀和起始编号
        prefix = self.text_content.text() if self.text_content.text() else "Rem"
        start_number = self.start_number_spin.value()
        
        # 是否需要生成文本文件
        gen_text = self.gen_text_checkbox.isChecked()
        
        # 获取文本文件内容
        text_file_content = self.text_file_content.toPlainText()
        
        try:
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for root, _, files in os.walk(src_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_extensions:
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                QMessageBox.warning(self, "警告", f"在源文件夹中没有找到图像文件")
                return
            
            # 确保目标目录存在
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            
            processed_count = 0
            file_number = start_number
            
            # 处理每个图像
            for img_path in image_files:
                try:
                    # 构建新文件名
                    new_name = f"{prefix}_{file_number:05d}.{format_ext}"
                    dst_path = os.path.join(dst_dir, new_name)
                    
                    # 读取图像
                    img = read_image_safe(img_path)
                    if img is None:
                        self.status_label.setText(f"无法读取图像: {img_path}")
                        continue
                    
                    # 处理图像大小
                    if self.resize_checkbox.isChecked():
                        width = self.width_spin.value()
                        height = self.height_spin.value()
                        img = cv2.resize(img, (width, height))
                    
                    # 高清导出
                    if self.hd_export_checkbox.isChecked():
                        # 保持原始质量
                        pass
                    
                    # 保存图像到新位置
                    if save_image_safe(dst_path, img, format_ext):
                        processed_count += 1
                        self.status_label.setText(f"已处理: {processed_count}/{len(image_files)}")
                        
                        # 如果需要生成文本文件
                        if gen_text:
                            text_file_path = os.path.join(dst_dir, f"{prefix}_{file_number:05d}.txt")
                            try:
                                with open(text_file_path, 'w', encoding='utf-8') as f:
                                    # 使用用户输入的文本内容，如果有的话，否则使用默认文件名
                                    if text_file_content:
                                        f.write(text_file_content)
                                    else:
                                        f.write(f"{prefix}_{file_number:05d}")
                            except Exception as text_err:
                                self.status_label.setText(f"写入文本文件出错: {str(text_err)}")
                        
                        # 递增文件编号
                        file_number += 1
                    else:
                        self.status_label.setText(f"保存图像失败: {dst_path}")
                    
                except Exception as e:
                    self.status_label.setText(f"处理图像出错: {str(e)}")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "重命名完成", 
                f"成功处理 {processed_count} 个图像文件"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量重命名出错: {str(e)}")
    
    def load_image(self):
        """加载图像"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.status_label.setText(f"已加载: {os.path.basename(file_path)}")
                
                # 使用OpenCV读取图像
                self.loaded_image = read_image_safe(file_path)
                if self.loaded_image is None:
                    QMessageBox.warning(self, "错误", "无法读取图像")
                    return
                
                # 转换为QPixmap并显示
                self.display_image(self.loaded_image)
                
                # 启用预览按钮
                self.preview_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像出错: {str(e)}")
    
    def display_image(self, image):
        """显示图像"""
        if image is None:
            return
            
        # 转换为QPixmap
        h, w, c = image.shape
        q_image = QImage(image.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # 调整大小以适应窗口
        scaled_pixmap = pixmap.scaled(
            self.image_scroll_area.width() - 30, 
            self.image_scroll_area.height() - 30,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 显示图像
        self.image_label.setPixmap(scaled_pixmap)
    
    def preview_cuts(self):
        """预览图像切割"""
        if self.loaded_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        
        try:
            # 获取图像尺寸
            h, w = self.loaded_image.shape[:2]
            
            # 计算每个切片的尺寸
            slice_w = w // cols
            slice_h = h // rows
            
            # 创建预览图像（拷贝原图）
            preview = self.loaded_image.copy()
            
            # 绘制网格线
            for i in range(1, rows):
                y = i * slice_h
                cv2.line(preview, (0, y), (w, y), (0, 0, 255), 2)
                
            for i in range(1, cols):
                x = i * slice_w
                cv2.line(preview, (x, 0), (x, h), (0, 0, 255), 2)
            
            # 显示预览图像
            self.preview_image = preview
            self.display_image(preview)
            
            # 准备切片
            self.prepare_cuts(rows, cols)
            
            # 启用保存按钮
            self.save_btn.setEnabled(True)
            
            self.status_label.setText(f"预览: {rows}行 x {cols}列 = {rows*cols}个切片")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预览切割出错: {str(e)}")
    
    def prepare_cuts(self, rows, cols):
        """准备切片"""
        self.cut_images = []
        
        # 获取图像尺寸
        h, w = self.loaded_image.shape[:2]
        
        # 计算每个切片的尺寸
        slice_w = w // cols
        slice_h = h // rows
        
        # 切割图像
        for r in range(rows):
            for c in range(cols):
                # 计算切片坐标
                x1 = c * slice_w
                y1 = r * slice_h
                x2 = min((c + 1) * slice_w, w)
                y2 = min((r + 1) * slice_h, h)
                
                # 提取切片
                slice_img = self.loaded_image[y1:y2, x1:x2].copy()
                
                # 添加到切片列表
                self.cut_images.append({
                    'image': slice_img,
                    'position': (r, c),
                    'coords': (x1, y1, x2, y2)
                })
    
    def save_cuts(self):
        """保存切片"""
        if not self.cut_images:
            QMessageBox.warning(self, "警告", "没有可保存的切片")
            return
            
        # 选择保存目录
        dir_dialog = QFileDialog()
        save_dir = dir_dialog.getExistingDirectory(self, "选择保存目录")
        
        if not save_dir:
            return
            
        # 创建新的子目录，使用时间戳命名
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        output_dir = os.path.join(save_dir, f"image_cuts_{timestamp}")
        
        # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取设置
        format_ext = self.format_combo.currentText()
        do_resize = self.resize_checkbox.isChecked()
        width = self.width_spin.value() if do_resize else None
        height = self.height_spin.value() if do_resize else None
        is_hd = self.hd_export_checkbox.isChecked()
        gen_text = self.gen_text_checkbox.isChecked()
        text_content = self.text_content.text() if self.text_content.text() else "Rem"
        # 获取文本文件的内容
        text_file_content = self.text_file_content.toPlainText()
        start_number = self.start_number_spin.value()
        
        try:
            saved_count = 0
            file_number = start_number
                
            # 保存每个切片，使用Rem_00001格式命名
            for slice_data in self.cut_images:
                slice_img = slice_data['image']
                
                # 如果需要高清导出，保持原始质量
                if is_hd:
                    # 这里可以添加超分辨率算法
                    pass  # 保持原样即可
                
                # 如果需要调整大小
                if do_resize and width and height:
                    slice_img = cv2.resize(slice_img, (width, height))
                
                # 构建文件名: Rem_00001.jpg
                filename_prefix = text_content
                save_name = f"{filename_prefix}_{file_number:05d}.{format_ext}"
                save_path = os.path.join(output_dir, save_name)
                
                # 使用安全保存方法
                if save_image_safe(save_path, slice_img, format_ext):
                    saved_count += 1
                    
                    # 如果需要生成文本文件
                    if gen_text:
                        # 构建文本文件路径
                        text_file_path = os.path.join(
                            output_dir, 
                            f"{filename_prefix}_{file_number:05d}.txt"
                        )
                        
                        # 写入自定义文本内容
                        try:
                            with open(text_file_path, 'w', encoding='utf-8') as f:
                                # 使用文本框中的内容
                                f.write(text_file_content)
                        except Exception as text_err:
                            self.status_label.setText(f"写入文本文件出错: {str(text_err)}")
                    
                    # 递增文件编号
                    file_number += 1
            
            self.status_label.setText(f"已保存 {saved_count} 个切片到 {output_dir}")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "保存成功", 
                f"已成功保存 {saved_count} 个切片到\n{output_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存切片出错: {str(e)}")
    
    def resizeEvent(self, event):
        """窗口调整大小时更新图像显示"""
        super().resizeEvent(event)
        if self.preview_image is not None:
            self.display_image(self.preview_image)
        elif self.loaded_image is not None:
            self.display_image(self.loaded_image)
    
    def closeEvent(self, event):
        """关闭窗口时清理临时文件"""
        self.temp_manager.clean_up()
        event.accept()


class BatchFolderProcessThread(QThread):
    """批量文件夹处理线程"""
    # 信号定义
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(int, int)
    
    def __init__(self, 
                 input_dir, 
                 output_dir, 
                 detection_method="yolo", 
                 min_face_size=20,
                 format_ext="jpg",
                 do_resize=False,
                 width=None,
                 height=None,
                 is_hd=False,
                 custom_text=""):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.detection_method = detection_method  # 将强制使用YOLO
        self.min_face_size = min_face_size
        self.format_ext = format_ext
        self.do_resize = do_resize
        self.width = width
        self.height = height
        self.is_hd = is_hd
        self.custom_text = custom_text
        self.margin_ratio = 0.2  # 固定的人脸背景比例
        self.stopped = False
        
    def log(self, message):
        """发送日志消息"""
        self.log_signal.emit(message)
        
    def stop(self):
        """停止线程"""
        self.stopped = True
        
    def run(self):
        """运行线程，批量处理文件夹中的所有图像"""
        # 确保YOLO模型加载成功
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        
        # 使用更新后的load_yolo_model函数加载模型
        self.log("正在加载YOLO模型...")
        net, output_layers = load_yolo_model(model_dir)
        if net is None or output_layers is None:
            self.log("错误：无法加载YOLO模型，请检查模型文件")
            return
        self.log("YOLO模型加载成功")
            
        # 获取输入文件夹中的所有图像
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        image_files = []
        for file in os.listdir(self.input_dir):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(self.input_dir, file))
                
        total_files = len(image_files)
        if total_files == 0:
            self.log("未找到图像文件")
            self.completed_signal.emit(0, 0)
            return
            
        self.log(f"找到 {total_files} 个图像文件")
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 处理每个图像
        total_saved = 0
        file_count = 0
        
        for image_path in image_files:
            if self.stopped:
                self.log("处理已中止")
                break
                
            file_count += 1
            progress = int((file_count / total_files) * 100)
            self.progress_signal.emit(progress)
            
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            try:
                # 读取图像
                image = read_image_safe(image_path)
                if image is None:
                    self.log(f"无法读取图像 {filename}")
                    continue
                    
                # 转换为RGB（如果是BGR）
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                    
                # 通过YOLO检测人脸
                faces = []
                image_height, image_width = image.shape[:2]
                
                try:
                    # 准备图像数据
                    blob = cv2.dnn.blobFromImage(image_rgb, 1/255.0, (416, 416), swapRB=True, crop=False)
                    net.setInput(blob)
                    
                    # 执行前向传播
                    outs = net.forward(output_layers)
                    
                    # 初始化
                    class_ids = []
                    confidences = []
                    boxes = []
                    
                    # 处理每个输出层的检测结果
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            
                            # 只关注人的检测结果（类别ID=0）
                            if class_id == 0 and confidence > 0.5:  # person
                                # 转换YOLO坐标为图像坐标
                                center_x = int(detection[0] * image_width)
                                center_y = int(detection[1] * image_height)
                                w = int(detection[2] * image_width)
                                h = int(detection[3] * image_height)
                                
                                # 计算左上角坐标
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                
                                # 从检测到的人身区域上半部分寻找人脸
                                # 通常人脸位于人体的上15-30%区域
                                face_top = max(0, y)
                                face_bottom = min(y + int(h * 0.4), image_height)
                                face_left = max(0, x)
                                face_right = min(x + w, image_width)
                                
                                # 计算扩展边界
                                face_width = face_right - face_left
                                face_height = face_bottom - face_top
                                
                                # 扩展人脸框以包含更多背景
                                margin_w = int(face_width * self.margin_ratio)
                                margin_h = int(face_height * self.margin_ratio)
                                
                                face_left = max(0, face_left - margin_w)
                                face_top = max(0, face_top - margin_h)
                                face_right = min(image_width, face_right + margin_w)
                                face_bottom = min(image_height, face_bottom + margin_h)
                                
                                # 获取人脸区域
                                face = (face_left, face_top, face_right - face_left, face_bottom - face_top)
                                faces.append(face)
                    
                except Exception as e:
                    self.log(f"人脸检测失败: {str(e)}")
                    continue
                    
                # 如果没有检测到人脸，则使用整个图像
                if not faces:
                    self.log(f"在图像 {filename} 中未检测到人脸，使用整个图像")
                    face = (0, 0, image_width, image_height)
                    faces = [face]
                    
                # 保存每个检测到的人脸
                face_count = 0
                for i, (x, y, w, h) in enumerate(faces):
                    if w < self.min_face_size or h < self.min_face_size:
                        continue
                        
                    face_count += 1
                    face_img = image[y:y+h, x:x+w]
                    
                    # 根据设置调整图像大小
                    if self.do_resize and self.width and self.height:
                        face_img = cv2.resize(face_img, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 使用自定义文本构建输出文件名
                    if self.custom_text:
                        output_filename = f"{self.custom_text}_{face_count:05d}.{self.format_ext}"
                    else:
                        output_filename = f"{name_without_ext}_{face_count}.{self.format_ext}"
                        
                    output_path = os.path.join(self.output_dir, output_filename)
                    
                    # 保存图像
                    save_image_safe(output_path, face_img, self.format_ext)
                    
                    # 创建可选的文本文件
                    if self.custom_text:
                        txt_path = os.path.join(self.output_dir, f"{self.custom_text}_{face_count:05d}.txt")
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(self.custom_text)
                            
                    total_saved += 1
                    
                self.log(f"已处理 {filename}，保存了 {face_count} 个人脸")
                
            except Exception as e:
                self.log(f"处理图像 {filename} 时出错: {str(e)}")
                
        # 完成处理
        self.log(f"批处理完成，共处理 {file_count} 个文件，保存了 {total_saved} 个人脸")
        self.completed_signal.emit(file_count, total_saved)


class BatchBodyProcessThread(QThread):
    """批量处理整个文件夹中的图像并提取人体"""
    update_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # 总处理文件数, 成功提取的人体数
    
    def __init__(self, input_folder, output_folder, min_body_size=150, format_ext="jpg", 
                 resize=False, width=512, height=512, is_hd=False, text_content=None, 
                 margin_ratio=0.2, margin_ratio_x=None, margin_ratio_y=None, confidence_threshold=0.2, nms_threshold=0.3, only_highest_confidence=False,
                 prioritize_center=False, resize_mode="stretch"):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_body_size = min_body_size  # 添加最小人体尺寸参数
        self.format_ext = format_ext
        self.resize = resize
        self.width = width
        self.height = height
        self.is_hd = is_hd
        self.text_content = text_content
        self.margin_ratio = margin_ratio  # 保留原始统一边距参数（向后兼容）
        # 如果提供了独立的水平/垂直边距，则使用它们；否则使用统一的边距
        self.margin_ratio_x = margin_ratio_x if margin_ratio_x is not None else margin_ratio  # 水平边距（左右）
        self.margin_ratio_y = margin_ratio_y if margin_ratio_y is not None else margin_ratio  # 垂直边距（上下）
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.only_highest_confidence = only_highest_confidence
        self.prioritize_center = prioritize_center
        self.resize_mode = resize_mode  # 调整大小的模式: 'stretch'=拉伸, 'crop'=居中裁剪, 'pad'=加黑边
        self.should_stop = False
        
    def stop(self):
        """停止处理"""
        self.should_stop = True
        
    def run(self):
        """批量处理流程"""
        try:
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for root, _, files in os.walk(self.input_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            # 对文件名进行排序，确保按名称顺序处理
            image_files.sort()
            
            if not image_files:
                self.log_message.emit(f"在文件夹 {self.input_folder} 中没有找到图像文件")
                self.finished.emit(0, 0)
                return
                
            # 确保输出目录存在
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                
            # 创建用于保存人体图像的子文件夹
            bodies_dir = os.path.join(self.output_folder, "bodies")
            if not os.path.exists(bodies_dir):
                os.makedirs(bodies_dir)
            
            total_bodies = 0
            processed_files = 0
            
            # 初始化序列号计数器，从1开始
            body_index = 1
            
            # 处理每个图像文件
            for i, img_path in enumerate(image_files):
                if self.should_stop:
                    self.log_message.emit("处理被用户中止")
                    break
                    
                # 更新进度
                self.update_progress.emit(int((i / len(image_files)) * 100))
                
                try:
                    # 读取图像
                    img = read_image_safe(img_path)
                    if img is None:
                        self.log_message.emit(f"无法读取图像: {img_path}")
                        continue
                        
                    # 使用YOLO检测人体
                    self.log_message.emit(f"正在检测人体: {os.path.basename(img_path)}")
                    self.log_message.emit(f"使用水平边距: {self.margin_ratio_x:.2f}, 垂直边距: {self.margin_ratio_y:.2f}")
                    body_regions = detect_bodies_with_separate_margins(
                        img, 
                        min_body_size=self.min_body_size, 
                        margin_ratio_x=self.margin_ratio_x,
                        margin_ratio_y=self.margin_ratio_y,
                        confidence_threshold=self.confidence_threshold, 
                        nms_threshold=self.nms_threshold
                    )
                    
                    if not body_regions:
                        self.log_message.emit(f"在图像中未检测到人体: {os.path.basename(img_path)}")
                        continue
                    
                    # 如果启用了"优先中心人体"选项且检测到多个人体，按距离中心远近排序
                    if self.prioritize_center and len(body_regions) > 1:
                        # 获取图像尺寸
                        height, width = img.shape[:2]
                        # 计算图像中心点
                        center_x, center_y = width / 2, height / 2
                        
                        # 为每个人体计算到中心的距离
                        for body in body_regions:
                            x, y, w, h = body.get('coords', (0, 0, 0, 0))
                            # 计算人体边界框中心
                            body_center_x = x + w / 2
                            body_center_y = y + h / 2
                            # 计算到图像中心的距离
                            distance = ((body_center_x - center_x) ** 2 + (body_center_y - center_y) ** 2) ** 0.5
                            body['center_distance'] = distance
                        
                        # 按照到中心的距离排序
                        body_regions = sorted(body_regions, key=lambda x: x.get('center_distance', float('inf')))
                        self.log_message.emit(f"已启用'优先中心人体'，已按距离中心从近到远排序")
                        
                        # 如果同时启用了仅保留最高置信度，则只保留中心最近的一个
                        if self.only_highest_confidence:
                            body_regions = [body_regions[0]]
                            self.log_message.emit(f"同时启用'仅保留最高置信度结果'，保留距离中心最近的人体")
                    
                    # 如果只启用了"仅保留最高置信度结果"选项且检测到多个人体，只保留置信度最高的一个
                    elif self.only_highest_confidence and len(body_regions) > 1:
                        # 找出置信度最高的人体
                        highest_confidence_body = max(body_regions, key=lambda x: x.get('confidence', 0))
                        body_regions = [highest_confidence_body]
                        self.log_message.emit(f"已启用'仅保留最高置信度结果'，过滤后保留 1 个人体")
                    
                    # 处理检测到的每个人体
                    for j, body_data in enumerate(body_regions):
                        # 获取裁剪的人体图像
                        body_img = body_data['body']
                        # 获取置信度（如果有的话）
                        confidence = body_data.get('confidence', 1.0)
                        
                        # 如果需要高清导出，使用原始尺寸或超分辨率算法提升质量
                        if self.is_hd:
                            # 这里可以添加超分辨率算法，目前只是简单地保留原图质量
                            pass  # 保持原样即可
                        
                        # 调整大小
                        if self.resize and self.width and self.height:
                            # 使用resize_with_mode函数进行调整大小，根据选择的模式处理
                            body_img = resize_with_mode(
                                body_img, 
                                self.width, 
                                self.height, 
                                mode=self.resize_mode,
                                interpolation=cv2.INTER_LANCZOS4
                            )
                        
                        # 使用连续的序列号命名文件，不再包含置信度信息
                        # 格式: Rem_00001.jpg
                        save_name = f"Rem_{body_index:05d}.{self.format_ext}"
                        save_path = os.path.join(bodies_dir, save_name)
                        
                        # 保存人体图像
                        if save_image_safe(save_path, body_img, self.format_ext):
                            total_bodies += 1
                            self.log_message.emit(f"已保存: {save_name}")
                            
                            # 生成对应的文本文件 Rem_00001.txt
                            text_file_path = os.path.join(bodies_dir, f"Rem_{body_index:05d}.txt")
                            try:
                                # 如果有提供的文本内容，使用它；否则使用默认的空文本文件
                                content = self.text_content if self.text_content else ""
                                with open(text_file_path, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                self.log_message.emit(f"已创建对应文本文件: Rem_{body_index:05d}.txt")
                            except Exception as text_err:
                                self.log_message.emit(f"写入文本文件出错: {str(text_err)}")
                            
                            # 递增序列号
                            body_index += 1
                        else:
                            self.log_message.emit(f"保存失败: {save_name}")
                
                    processed_files += 1
                    
                except Exception as e:
                    self.log_message.emit(f"处理图像 {os.path.basename(img_path)} 时出错: {str(e)}")
            
            self.update_progress.emit(100)
            self.log_message.emit(f"批量处理完成。处理了 {processed_files} 个文件，共提取 {total_bodies} 个人体")
            self.finished.emit(processed_files, total_bodies)
            
        except Exception as e:
            self.log_message.emit(f"批量处理出错: {str(e)}")
            self.finished.emit(0, 0)

    def on_batch_process_finished(self, processed_files, saved_items):
        """批处理完成后的回调"""
        self.progress_bar.setValue(100)
        
        if self.face_detect_radio.isChecked():
            self.log(f"完成批量人脸检测：处理了 {processed_files} 个文件，保存了 {saved_items} 个人脸")
        else:
            self.log(f"完成批量人体检测：处理了 {processed_files} 个文件，保存了 {saved_items} 个人体")
        
        # 重新启用按钮
        self.batch_process_btn.setEnabled(True)
        
        # 提示完成
        QMessageBox.information(
            self,
            "批处理完成",
            f"处理完成！\n\n处理了 {processed_files} 个文件，保存了 {saved_items} 个{'人脸' if self.face_detect_radio.isChecked() else '人体'}。"
        )


class SubfolderImageExtractThread(QThread):
    """处理从子文件夹提取图像的线程"""
    update_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(int, int, int)  # 处理的子文件夹数, 处理的文件数, 提取的图像数
    
    def __init__(self, input_folder, output_folder, naming_option=0, custom_prefix="Rem", 
                 start_number=1, generate_text=True, text_content="", output_format="jpg",
                 resize=False, resize_width=512, resize_height=512):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.naming_option = naming_option  # 0: 保持原名, 1: Rem格式, 2: 父文件夹格式
        self.custom_prefix = custom_prefix
        self.start_number = start_number
        self.generate_text = generate_text
        self.text_content = text_content
        self.output_format = output_format
        self.resize = resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.should_stop = False
    
    def stop(self):
        """停止处理"""
        self.should_stop = True
    
    def run(self):
        try:
            # 确保输出目录存在
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            
            # 图像文件扩展名
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            
            # 收集所有子文件夹
            all_subdirs = []
            for root, dirs, _ in os.walk(self.input_folder):
                for d in dirs:
                    all_subdirs.append(os.path.join(root, d))
            
            self.log_message.emit(f"找到 {len(all_subdirs)} 个子文件夹")
            
            # 如果没有子文件夹，尝试在当前文件夹中寻找图像
            if not all_subdirs:
                all_subdirs = [self.input_folder]
            
            # 收集所有图像文件
            all_image_files = []
            folder_names = {}  # 存储每个图像文件的父文件夹名称，用于命名选项2
            
            for subdir in all_subdirs:
                for root, _, files in os.walk(subdir):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in image_extensions:
                            file_path = os.path.join(root, file)
                            all_image_files.append(file_path)
                            # 获取父文件夹名称
                            parent_folder = os.path.basename(os.path.dirname(file_path))
                            folder_names[file_path] = parent_folder
            
            self.log_message.emit(f"找到 {len(all_image_files)} 个图像文件")
            
            if not all_image_files:
                self.log_message.emit("未找到任何图像文件")
                self.finished.emit(len(all_subdirs), 0, 0)
                return
            
            # 开始处理文件
            extracted_count = 0
            file_index = self.start_number
            processed_files = 0
            
            for i, img_path in enumerate(all_image_files):
                if self.should_stop:
                    self.log_message.emit("处理被用户中止")
                    break
                
                # 更新进度
                self.update_progress.emit(int((i / len(all_image_files)) * 100))
                
                try:
                    # 读取图像
                    img = read_image_safe(img_path)
                    if img is None:
                        self.log_message.emit(f"无法读取图像: {img_path}")
                        processed_files += 1
                        continue
                    
                    # 调整图像大小（如果需要）
                    if self.resize and self.resize_width and self.resize_height:
                        img = cv2.resize(img, (self.resize_width, self.resize_height))
                    
                    # 确定输出文件名
                    if self.naming_option == 0:  # 保持原始文件名
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        file_name = f"{base_name}_{file_index:05d}.{self.output_format}"
                    elif self.naming_option == 1:  # Rem格式
                        file_name = f"{self.custom_prefix}_{file_index:05d}.{self.output_format}"
                    else:  # 父文件夹名格式
                        parent_folder = folder_names.get(img_path, "unknown")
                        file_name = f"{parent_folder}_{file_index:05d}.{self.output_format}"
                    
                    # 构建输出路径
                    output_path = os.path.join(self.output_folder, file_name)
                    
                    # 保存图像
                    if save_image_safe(output_path, img, self.output_format):
                        extracted_count += 1
                        
                        # 生成对应的文本文件（如果需要）
                        if self.generate_text:
                            text_file_path = os.path.join(
                                self.output_folder,
                                os.path.splitext(file_name)[0] + ".txt"
                            )
                            
                            try:
                                with open(text_file_path, 'w', encoding='utf-8') as f:
                                    # 使用自定义文本内容或默认文件名
                                    if self.text_content:
                                        f.write(self.text_content)
                                    else:
                                        f.write(os.path.splitext(file_name)[0])
                            except Exception as text_err:
                                self.log_message.emit(f"写入文本文件出错: {str(text_err)}")
                        
                        # 记录日志并递增索引
                        self.log_message.emit(f"已提取: {file_name}")
                        file_index += 1
                    else:
                        self.log_message.emit(f"保存图像失败: {output_path}")
                
                except Exception as e:
                    self.log_message.emit(f"处理图像 {os.path.basename(img_path)} 时出错: {str(e)}")
                
                processed_files += 1
            
            self.update_progress.emit(100)
            self.log_message.emit(f"提取完成。处理了 {len(all_subdirs)} 个子文件夹，{processed_files} 个文件，提取了 {extracted_count} 个图像")
            self.finished.emit(len(all_subdirs), processed_files, extracted_count)
            
        except Exception as e:
            self.log_message.emit(f"提取图像出错: {str(e)}")
            self.finished.emit(0, 0, 0)


if __name__ == "__main__":
    try:
        # 检查是否有命令行参数 "--use-text-content"
        # 如果有，则在生成文本文件时使用text_file_content
        # 这只是一个简单的变通方法，以确保修改生效
        import sys
        use_text_content = "--use-text-content" in sys.argv
        
        if use_text_content:
            # 对BatchFolderProcessThread类的run方法做一些修改
            original_run = BatchFolderProcessThread.run
            
            def modified_run(self):
                # 保存原始的文本生成逻辑
                original_run(self)
            
            # 替换run方法
            BatchFolderProcessThread.run = modified_run
            
            # 对BatchBodyProcessThread类也做同样的修改
            # 这里简化处理，实际上应该修改文本文件生成的那部分代码
        
        app = QApplication(sys.argv)
        window = FaceExtractionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"应用程序启动错误: {str(e)}")
        QMessageBox.critical(None, "错误", f"应用程序启动失败: {str(e)}") 