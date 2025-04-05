#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import uuid
import shutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLineEdit, QCheckBox, QSlider, QSpinBox,
    QRadioButton, QProgressBar, QTextEdit, QGroupBox, QMessageBox, QGridLayout,
    QListWidget, QSplitter, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

##############################
# 工具函数：图像读取 & 保存 & 模型加载
##############################

def read_image_safe(file_path):
    """安全地读取图像，支持中文路径"""
    try:
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            # 再尝试OpenCV直接读
            img = cv2.imread(file_path)
        return img
    except Exception as e:
        print(f"读取图像出错: {str(e)}")
        return None

def save_image_safe(file_path, image, ext='.jpg'):
    """安全地保存图像，支持中文路径"""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        _, buf = cv2.imencode(ext, image)
        with open(file_path, 'wb') as f:
            f.write(buf)
        return True
    except Exception as e:
        print(f"保存图像出错: {str(e)}")
        return False

def load_yolo_model(model_dir="models", temp_dir="C:\\temp_models"):
    """加载YOLOv3模型，兼容中文路径"""
    try:
        weights_path = os.path.join(model_dir, "yolov3.weights")
        cfg_path = os.path.join(model_dir, "yolov3.cfg")

        # 如果路径里有中文，OpenCV可能会打不开，所以尝试复制到临时英文路径
        # 也兼容了OpenCV某些版本会要求ASCII路径
        if not os.path.exists(weights_path) or not os.path.exists(cfg_path):
            print("模型文件不存在，请检查 yolo 模型！")
            return None, None

        try:
            net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        except Exception as e:
            print(f"直接加载模型失败: {str(e)}\n尝试复制到临时路径加载...")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_weights = os.path.join(temp_dir, "yolov3.weights")
            temp_cfg = os.path.join(temp_dir, "yolov3.cfg")
            shutil.copy2(weights_path, temp_weights)
            shutil.copy2(cfg_path, temp_cfg)
            net = cv2.dnn.readNetFromDarknet(temp_cfg, temp_weights)

        # 获取输出层
        layer_names = net.getLayerNames()
        out_layers_indices = net.getUnconnectedOutLayers()
        out_layers_indices = out_layers_indices.flatten() if hasattr(out_layers_indices, 'flatten') else out_layers_indices
        output_layers = [layer_names[i - 1] for i in out_layers_indices]
        return net, output_layers
    except Exception as e:
        print(f"加载YOLO模型出错: {str(e)}")
        return None, None

# 修改：添加保持纵横比的缩放函数
def resize_keep_aspect_ratio(image, target_width, target_height):
    """保持纵横比缩放图像到指定尺寸，不足部分填充黑色"""
    h, w = image.shape[:2]
    target_ratio = target_width / target_height
    img_ratio = w / h

    # 确定缩放因子，使图像适应目标尺寸
    if img_ratio > target_ratio:
        # 图像更宽，按宽度缩放
        new_w = target_width
        new_h = int(new_w / img_ratio)
    else:
        # 图像更高，按高度缩放
        new_h = target_height
        new_w = int(new_h * img_ratio)
    
    # 缩放图像
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建黑色背景
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # 计算居中位置
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # 将缩放后的图像放在画布中央
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

##############################
# 线程：批量检测 - 人脸/上半身
##############################

class FaceBatchThread(QThread):
    """用YOLO批量检测人脸(或上半身)的线程"""
    update_progress = pyqtSignal(int)  # 进度
    log_signal = pyqtSignal(str)       # 日志
    finished_signal = pyqtSignal(int)  # 完成后传回"成功保存文件数"

    def __init__(self, input_dir, output_dir, min_face_size=80, margin_ratio=0.2,
                 confidence_threshold=0.3, do_resize=False, resize_w=512, resize_h=512,
                 is_hd=False, detect_upper_body=False, create_mirror=False, center_crop=False):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_face_size = min_face_size
        self.margin_ratio = margin_ratio
        self.confidence_threshold = confidence_threshold
        self.do_resize = do_resize
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.is_hd = is_hd
        self.detect_upper_body = detect_upper_body
        self.create_mirror = create_mirror  # 是否创建镜像图片
        self.center_crop = center_crop      # 是否使用居中裁切

        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        try:
            net, output_layers = load_yolo_model()
            if net is None or output_layers is None:
                self.log_signal.emit("YOLO模型加载失败，无法检测")
                self.finished_signal.emit(0)
                return

            # 遍历输入目录图像文件
            exts = ['.jpg', '.jpeg', '.png', '.bmp']
            all_files = [f for f in os.listdir(self.input_dir)
                         if os.path.splitext(f)[1].lower() in exts]
            total_files = len(all_files)
            if total_files == 0:
                self.log_signal.emit("未在输入目录找到图像文件")
                self.finished_signal.emit(0)
                return

            saved_count = 0
            for i, filename in enumerate(all_files):
                if self.stop_flag:
                    self.log_signal.emit("检测被停止")
                    break

                progress = int((i / total_files) * 100)
                self.update_progress.emit(progress)

                in_path = os.path.join(self.input_dir, filename)
                img = read_image_safe(in_path)
                if img is None:
                    self.log_signal.emit(f"无法读取图像: {filename}")
                    continue

                h, w = img.shape[:2]

                # 用YOLO检测 person
                blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416),
                                             swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                # 解析检测结果
                boxes = []
                confidences = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # YOLO person类别 id=0
                        if class_id == 0 and confidence > self.confidence_threshold:
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            box_w = int(detection[2] * w)
                            box_h = int(detection[3] * h)
                            x = int(center_x - box_w / 2)
                            y = int(center_y - box_h / 2)
                            boxes.append([x, y, box_w, box_h])
                            confidences.append(float(confidence))

                # 如果没检测到 person，则跳过or整图当脸
                if not boxes:
                    self.log_signal.emit(f"[{filename}] 未检测到 person，使用整图")
                    x, y, box_w, box_h = 0, 0, w, h
                    boxes = [[x, y, box_w, box_h]]
                    confidences = [1.0]

                # NMS 去重
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
                if len(indices) == 0:
                    self.log_signal.emit(f"[{filename}] 检测到 person, 但被NMS过滤光了")
                    continue

                # 只取NMS过滤后保留的框
                final_boxes = [boxes[i[0]] if isinstance(i, list) else boxes[i] for i in indices]

                # 提取区域
                for idx, (x, y, box_w, box_h) in enumerate(final_boxes):
                    # 如果要只取**上半身**(detect_upper_body=True)，
                    # 人脸大概在上半部分0.4~0.5
                    if self.detect_upper_body:
                        # 只截取这个person框的上半部分
                        head_h = int(box_h * 0.4)  # 可调
                        face_y = max(0, y)
                        face_h = min(head_h, (y + box_h) - face_y)
                    else:
                        # 如果仅做人脸的话，可以自行对"上半身"定义
                        # 这里直接把整个人作为"脸" :)
                        face_y = y
                        face_h = box_h

                    # 防止越界
                    face_x = max(0, x)
                    face_h = max(0, face_h)
                    face_w = box_w

                    # 扩展边缘 margin
                    margin_w = int(face_w * self.margin_ratio)
                    margin_h = int(face_h * self.margin_ratio)

                    face_x_expanded = max(0, face_x - margin_w)
                    face_y_expanded = max(0, face_y - margin_h)
                    face_w_expanded = min(w - face_x_expanded, face_w + 2 * margin_w)
                    face_h_expanded = min(h - face_y_expanded, face_h + 2 * margin_h)

                    # 如果最终区域比min_face_size要小，跳过
                    if face_w_expanded < self.min_face_size or face_h_expanded < self.min_face_size:
                        self.log_signal.emit(f"[{filename}] 检测框太小，跳过")
                        continue

                    # 裁剪
                    if self.center_crop:
                        # 确保自定义尺寸已设置
                        if not self.do_resize:
                            # 如果没有勾选自定义尺寸，默认设置为正方形
                            target_w = target_h = max(face_w_expanded, face_h_expanded)
                        else:
                            # 使用自定义尺寸
                            target_w = self.resize_w
                            target_h = self.resize_h
                        
                        # 居中裁切：计算人物中心点
                        person_center_x = x + box_w // 2
                        person_center_y = face_y + face_h // 2
                        
                        # 计算裁切区域，确保以人物为中心
                        crop_x = max(0, person_center_x - target_w // 2)
                        crop_y = max(0, person_center_y - target_h // 2)
                        
                        # 防止越界
                        if crop_x + target_w > w:
                            crop_x = max(0, w - target_w)
                        if crop_y + target_h > h:
                            crop_y = max(0, h - target_h)
                        
                        # 调整裁切大小，如果图像小于目标尺寸
                        actual_w = min(target_w, w - crop_x)
                        actual_h = min(target_h, h - crop_y)
                        
                        # 裁切
                        cropped = img[crop_y:crop_y+actual_h, crop_x:crop_x+actual_w]
                        
                        # 如果实际裁切尺寸与目标尺寸不一致，需要调整
                        if actual_w != target_w or actual_h != target_h:
                            # 创建一个黑色背景
                            final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                            # 将裁切的图像放在中央
                            y_offset = (target_h - actual_h) // 2
                            x_offset = (target_w - actual_w) // 2
                            final_img[y_offset:y_offset+actual_h, x_offset:x_offset+actual_w] = cropped
                            cropped = final_img
                    else:
                        # 原始裁切方式
                        cropped = img[face_y_expanded:face_y_expanded+face_h_expanded,
                                      face_x_expanded:face_x_expanded+face_w_expanded]

                    # 修改：更改调整大小的逻辑，保持纵横比
                    if self.do_resize:
                        cropped = resize_keep_aspect_ratio(cropped, self.resize_w, self.resize_h)

                    # 生成唯一文件名
                    base_name = os.path.splitext(filename)[0]
                    unique = uuid.uuid4().hex[:6]
                    out_name = f"{base_name}_face_{idx+1}_{unique}.jpg"
                    out_path = os.path.join(self.output_dir, out_name)

                    if save_image_safe(out_path, cropped, '.jpg'):
                        saved_count += 1
                        self.log_signal.emit(f"已保存: {out_name}")
                        
                        # 添加：如果需要镜像，创建并保存镜像图片
                        if self.create_mirror:
                            mirror_img = cv2.flip(cropped, 1)  # 水平翻转
                            mirror_name = f"{base_name}_face_{idx+1}_{unique}_mirror.jpg"
                            mirror_path = os.path.join(self.output_dir, mirror_name)
                            if save_image_safe(mirror_path, mirror_img, '.jpg'):
                                saved_count += 1
                                self.log_signal.emit(f"已保存镜像: {mirror_name}")
                            else:
                                self.log_signal.emit(f"保存镜像失败: {mirror_name}")
                    else:
                        self.log_signal.emit(f"保存失败: {out_name}")

            # 批量结束
            self.update_progress.emit(100)
            self.log_signal.emit(f"处理完成，共保存 {saved_count} 个结果图")
            self.finished_signal.emit(saved_count)
        except Exception as e:
            self.log_signal.emit(f"批量检测时出错: {str(e)}")
            self.finished_signal.emit(0)


##############################
# 主窗口：GUI
##############################

class FaceBatchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("批量人脸(上半身)检测 - YOLOv3")
        self.setGeometry(100, 100, 800, 600)  # 增大窗口尺寸

        self.thread = None
        self.current_preview_image = None  # 当前预览的图像
        self.preview_boxes = []  # 预览中的检测框

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 创建分割器，上部是控制区域，下部是预览区域
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 控制区域
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        splitter.addWidget(control_widget)

        # 输入输出目录
        dir_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.output_line = QLineEdit()
        self.input_line.setPlaceholderText("选择输入目录")
        self.output_line.setPlaceholderText("选择输出目录")
        input_btn = QPushButton("浏览(输入)")
        output_btn = QPushButton("浏览(输出)")
        input_btn.clicked.connect(self.browse_input)
        output_btn.clicked.connect(self.browse_output)

        dir_layout.addWidget(QLabel("输入目录:"))
        dir_layout.addWidget(self.input_line)
        dir_layout.addWidget(input_btn)

        dir_layout.addWidget(QLabel("输出目录:"))
        dir_layout.addWidget(self.output_line)
        dir_layout.addWidget(output_btn)

        control_layout.addLayout(dir_layout)

        # 参数设置
        param_group = QGroupBox("检测设置")
        param_layout = QGridLayout(param_group)

        # 最小尺寸
        param_layout.addWidget(QLabel("最小检测尺寸(像素):"), 0, 0)
        self.min_face_spin = QSpinBox()
        self.min_face_spin.setRange(20, 2000)
        self.min_face_spin.setValue(80)
        param_layout.addWidget(self.min_face_spin, 0, 1)

        # 置信度阈值
        param_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 99)  # 对应 0.1 ~ 0.99
        self.conf_slider.setValue(30)      # 0.3
        self.conf_label = QLabel("0.30")
        param_layout.addWidget(self.conf_slider, 1, 1)
        param_layout.addWidget(self.conf_label, 1, 2)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)

        # 边距比例
        param_layout.addWidget(QLabel("边距比例:"), 2, 0)
        self.margin_slider = QSlider(Qt.Horizontal)
        self.margin_slider.setRange(0, 50)  # 0 ~ 0.5
        self.margin_slider.setValue(20)     # 0.2
        self.margin_label = QLabel("0.20")
        param_layout.addWidget(self.margin_slider, 2, 1)
        param_layout.addWidget(self.margin_label, 2, 2)
        self.margin_slider.valueChanged.connect(self.on_margin_changed)

        # 是否只取上半身(检测头肩)
        self.upper_body_check = QCheckBox("只截取上半身区域(适合半身人像)")
        param_layout.addWidget(self.upper_body_check, 3, 0, 1, 3)
        
        # 添加: 是否创建镜像图片选项
        self.mirror_check = QCheckBox("同时创建镜像图片")
        param_layout.addWidget(self.mirror_check, 4, 0, 1, 3)

        # 添加: 居中裁切选项
        self.center_crop_check = QCheckBox("使用居中裁切(按自定义尺寸)")
        param_layout.addWidget(self.center_crop_check, 5, 0, 1, 3)
        
        # 关联居中裁切和自定义尺寸的状态
        self.center_crop_check.toggled.connect(lambda c: (
            self.resize_check.setChecked(c if c else self.resize_check.isChecked())
        ))

        control_layout.addWidget(param_group)

        # 调整大小
        resize_group = QGroupBox("输出尺寸")
        resize_layout = QHBoxLayout(resize_group)
        self.resize_check = QCheckBox("自定义尺寸")
        self.resize_check.setChecked(False)
        self.resize_w_spin = QSpinBox()
        self.resize_h_spin = QSpinBox()
        self.resize_w_spin.setRange(32, 4096)
        self.resize_h_spin.setRange(32, 4096)
        self.resize_w_spin.setValue(512)
        self.resize_h_spin.setValue(512)
        self.resize_w_spin.setEnabled(False)
        self.resize_h_spin.setEnabled(False)
        self.resize_check.toggled.connect(lambda c: (
            self.resize_w_spin.setEnabled(c),
            self.resize_h_spin.setEnabled(c)
        ))

        resize_layout.addWidget(self.resize_check)
        resize_layout.addWidget(QLabel("宽:"))
        resize_layout.addWidget(self.resize_w_spin)
        resize_layout.addWidget(QLabel("高:"))
        resize_layout.addWidget(self.resize_h_spin)

        control_layout.addWidget(resize_group)

        # 开始 & 停止按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止")
        self.preview_btn = QPushButton("预览检测结果")  # 添加预览按钮
        self.stop_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)  # 初始禁用预览按钮
        self.start_btn.clicked.connect(self.start_detect)
        self.stop_btn.clicked.connect(self.stop_detect)
        self.preview_btn.clicked.connect(self.preview_detection)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.preview_btn)  # 添加预览按钮到布局
        control_layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        # 日志输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        control_layout.addWidget(self.log_text)
        
        # 预览区域
        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)
        splitter.addWidget(preview_widget)
        
        # 左侧：文件列表
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        preview_layout.addWidget(self.file_list, 1)
        
        # 右侧：预览图
        self.preview_label = QLabel("选择图片进行预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f5f5f5;")
        preview_layout.addWidget(self.preview_label, 3)
        
        # 设置分割器的初始大小
        splitter.setSizes([400, 200])

    def on_conf_changed(self, value):
        v = value / 100.0
        self.conf_label.setText(f"{v:.2f}")

    def on_margin_changed(self, value):
        v = value / 100.0
        self.margin_label.setText(f"{v:.2f}")

    def browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if folder:
            self.input_line.setText(folder)
            self.update_file_list(folder)
            self.preview_btn.setEnabled(True)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.output_line.setText(folder)

    def update_file_list(self, folder):
        """更新文件列表"""
        self.file_list.clear()
        if not os.path.exists(folder):
            return
            
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for file in os.listdir(folder):
            if os.path.splitext(file)[1].lower() in exts:
                self.file_list.addItem(file)

    def on_file_selected(self, item):
        """当列表中的文件被选中时，显示预览"""
        if not item:
            return
            
        filename = item.text()
        input_dir = self.input_line.text().strip()
        if not input_dir or not os.path.exists(input_dir):
            return
            
        # 加载并显示图片
        img_path = os.path.join(input_dir, filename)
        img = read_image_safe(img_path)
        if img is not None:
            self.current_preview_image = img.copy()
            self.display_preview_image(img)
            self.log(f"已加载图片: {filename}")
        else:
            self.log(f"无法加载图片: {filename}")

    def display_preview_image(self, img, boxes=None):
        """在预览区域显示图片，可选择性显示检测框"""
        h, w = img.shape[:2]
        bytes_per_line = 3 * w
        
        # 转换为RGB显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 如果有检测框，绘制它们
        if boxes:
            for box in boxes:
                x, y, box_w, box_h = box
                cv2.rectangle(img_rgb, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        
        # 转换为QImage显示
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 计算适合预览区的大小
        preview_size = self.preview_label.size()
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.preview_label.setPixmap(scaled_pixmap)

    def preview_detection(self):
        """预览当前图片的检测结果"""
        if self.current_preview_image is None:
            QMessageBox.warning(self, "错误", "请先选择一张图片")
            return
            
        try:
            # 加载模型
            net, output_layers = load_yolo_model()
            if net is None or output_layers is None:
                self.log("YOLO模型加载失败，无法检测")
                return
                
            img = self.current_preview_image.copy()
            h, w = img.shape[:2]
            
            # 用YOLO检测
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            # 解析检测结果
            boxes = []
            confidences = []
            conf_threshold = self.conf_slider.value() / 100.0
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > conf_threshold:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        box_w = int(detection[2] * w)
                        box_h = int(detection[3] * h)
                        x = int(center_x - box_w / 2)
                        y = int(center_y - box_h / 2)
                        boxes.append([x, y, box_w, box_h])
                        confidences.append(float(confidence))
            
            # 如果没检测到，使用整图
            if not boxes:
                self.log("未检测到人物，将使用整图")
                boxes = [[0, 0, w, h]]
                confidences = [1.0]
                
            # NMS 去重
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
            if len(indices) == 0:
                self.log("检测结果被NMS过滤，无有效区域")
                return
                
            # 处理检测框
            final_boxes = [boxes[i[0]] if isinstance(i, list) else boxes[i] for i in indices]
            self.preview_boxes = final_boxes
            
            # 显示图片和框
            self.display_preview_image(img, self.preview_boxes)
            self.log(f"检测到 {len(self.preview_boxes)} 个区域")
            
            # 添加：显示预览提取结果
            self.show_extraction_preview(img, final_boxes)
            
        except Exception as e:
            self.log(f"预览检测出错: {str(e)}")
            
    def show_extraction_preview(self, img, boxes):
        """显示提取结果预览"""
        if not boxes:
            return
            
        # 创建预览窗口
        preview_window = QWidget(self)
        preview_window.setWindowTitle("提取结果预览")
        preview_layout = QVBoxLayout(preview_window)
        
        # 标题标签
        title_label = QLabel("以下是提取结果预览：")
        preview_layout.addWidget(title_label)
        
        # 创建滚动区域显示所有提取结果
        scroll_area = QWidget()
        scroll_layout = QHBoxLayout(scroll_area)
        
        # 处理每个检测框
        for idx, (x, y, box_w, box_h) in enumerate(boxes):
            # 应用当前参数
            margin_ratio = self.margin_slider.value() / 100.0
            detect_upper_body = self.upper_body_check.isChecked()
            min_face_size = self.min_face_spin.value()
            do_resize = self.resize_check.isChecked()
            resize_w = self.resize_w_spin.value()
            resize_h = self.resize_h_spin.value()
            create_mirror = self.mirror_check.isChecked()
            center_crop = self.center_crop_check.isChecked()  # 获取居中裁切设置
            
            h, w = img.shape[:2]
            
            # 如果要只取上半身
            if detect_upper_body:
                # 只截取上半部分
                head_h = int(box_h * 0.4)
                face_y = max(0, y)
                face_h = min(head_h, (y + box_h) - face_y)
            else:
                face_y = y
                face_h = box_h
            
            # 防止越界
            face_x = max(0, x)
            face_h = max(0, face_h)
            face_w = box_w
            
            # 扩展边缘 margin
            margin_w = int(face_w * margin_ratio)
            margin_h = int(face_h * margin_ratio)
            
            face_x_expanded = max(0, face_x - margin_w)
            face_y_expanded = max(0, face_y - margin_h)
            face_w_expanded = min(w - face_x_expanded, face_w + 2 * margin_w)
            face_h_expanded = min(h - face_y_expanded, face_h + 2 * margin_h)
            
            # 如果最终区域太小，跳过
            if face_w_expanded < min_face_size or face_h_expanded < min_face_size:
                continue
            
            # 裁剪 - 应用与FaceBatchThread一致的裁切逻辑
            if center_crop:
                # 确保自定义尺寸已设置
                if not do_resize:
                    # 如果没有勾选自定义尺寸，默认设置为正方形
                    target_w = target_h = max(face_w_expanded, face_h_expanded)
                else:
                    # 使用自定义尺寸
                    target_w = resize_w
                    target_h = resize_h
                
                # 居中裁切：计算人物中心点
                person_center_x = x + box_w // 2
                person_center_y = face_y + face_h // 2
                
                # 计算裁切区域，确保以人物为中心
                crop_x = max(0, person_center_x - target_w // 2)
                crop_y = max(0, person_center_y - target_h // 2)
                
                # 防止越界
                if crop_x + target_w > w:
                    crop_x = max(0, w - target_w)
                if crop_y + target_h > h:
                    crop_y = max(0, h - target_h)
                
                # 调整裁切大小，如果图像小于目标尺寸
                actual_w = min(target_w, w - crop_x)
                actual_h = min(target_h, h - crop_y)
                
                # 裁切
                cropped = img[crop_y:crop_y+actual_h, crop_x:crop_x+actual_w]
                
                # 如果实际裁切尺寸与目标尺寸不一致，需要调整
                if actual_w != target_w or actual_h != target_h:
                    # 创建一个黑色背景
                    final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    # 将裁切的图像放在中央
                    y_offset = (target_h - actual_h) // 2
                    x_offset = (target_w - actual_w) // 2
                    final_img[y_offset:y_offset+actual_h, x_offset:x_offset+actual_w] = cropped
                    cropped = final_img
            else:
                # 原始裁切方式
                cropped = img[face_y_expanded:face_y_expanded+face_h_expanded,
                            face_x_expanded:face_x_expanded+face_w_expanded]
            
            # 调整大小（保持纵横比）
            if do_resize:
                cropped = resize_keep_aspect_ratio(cropped, resize_w, resize_h)
                
            # 创建结果显示组
            result_group = QGroupBox(f"区域 {idx+1}")
            result_layout = QVBoxLayout(result_group)
            
            # 显示裁剪后的图像
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            h, w = cropped_rgb.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(cropped_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 限制预览图大小
            preview_size = 200
            scaled_pixmap = pixmap.scaled(preview_size, preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            img_label = QLabel()
            img_label.setPixmap(scaled_pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            result_layout.addWidget(img_label)
            
            # 如果有镜像选项，也显示镜像图
            if create_mirror:
                mirror_img = cv2.flip(cropped, 1)  # 水平翻转
                mirror_rgb = cv2.cvtColor(mirror_img, cv2.COLOR_BGR2RGB)
                h, w = mirror_rgb.shape[:2]
                bytes_per_line = 3 * w
                q_img = QImage(mirror_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(preview_size, preview_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                mirror_label = QLabel("镜像图")
                result_layout.addWidget(mirror_label)
                
                img_label = QLabel()
                img_label.setPixmap(scaled_pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                result_layout.addWidget(img_label)
            
            # 添加到滚动区域
            scroll_layout.addWidget(result_group)
        
        # 将滚动区域添加到主布局
        scroll_container = QScrollArea()
        scroll_container.setWidget(scroll_area)
        scroll_container.setWidgetResizable(True)
        scroll_container.setMinimumHeight(300)
        preview_layout.addWidget(scroll_container)
        
        # 确定按钮
        ok_button = QPushButton("确定")
        ok_button.clicked.connect(preview_window.close)
        preview_layout.addWidget(ok_button)
        
        # 显示窗口
        preview_window.setMinimumSize(600, 400)
        preview_window.show()

    def start_detect(self):
        input_dir = self.input_line.text().strip()
        output_dir = self.output_line.text().strip()
        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "错误", "输入目录无效")
            return
        if not output_dir:
            QMessageBox.warning(self, "错误", "输出目录不能为空")
            return

        min_face_size = self.min_face_spin.value()
        conf_threshold = self.conf_slider.value() / 100.0
        margin_ratio = self.margin_slider.value() / 100.0
        do_resize = self.resize_check.isChecked()
        rw = self.resize_w_spin.value()
        rh = self.resize_h_spin.value()
        detect_upper_body = self.upper_body_check.isChecked()
        create_mirror = self.mirror_check.isChecked()  # 获取是否创建镜像的选项
        center_crop = self.center_crop_check.isChecked()  # 获取是否使用居中裁切

        self.log_text.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.preview_btn.setEnabled(False)

        self.thread = FaceBatchThread(
            input_dir=input_dir,
            output_dir=output_dir,
            min_face_size=min_face_size,
            margin_ratio=margin_ratio,
            confidence_threshold=conf_threshold,
            do_resize=do_resize,
            resize_w=rw,
            resize_h=rh,
            is_hd=False,
            detect_upper_body=detect_upper_body,
            create_mirror=create_mirror,  # 传递镜像选项
            center_crop=center_crop  # 传递居中裁切选项
        )

        self.thread.update_progress.connect(self.progress_bar.setValue)
        self.thread.log_signal.connect(self.log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def stop_detect(self):
        if self.thread and self.thread.isRunning():
            self.log("停止检测中...")
            self.thread.stop()
            self.stop_btn.setEnabled(False)

    def on_finished(self, saved_count):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "完成", f"处理完成，共保存 {saved_count} 个结果图")

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


def main():
    app = QApplication(sys.argv)
    window = FaceBatchApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 