#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                            QTextEdit, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class TextCleanerThread(QThread):
    """处理文本文件清理的线程"""
    update_progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int)  # 总处理文件数, 成功删除的文件数
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.should_stop = False
    
    def stop(self):
        """停止线程"""
        self.should_stop = True
    
    def run(self):
        """运行线程，清理所有文本文件"""
        if not os.path.exists(self.folder_path):
            self.log_message.emit(f"错误: 目录 {self.folder_path} 不存在")
            self.finished_signal.emit(0, 0)
            return
            
        self.log_message.emit(f"开始扫描目录: {self.folder_path}")
        
        # 收集所有文本文件
        txt_files = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(".txt"):
                    txt_files.append(os.path.join(root, file))
        
        total_files = len(txt_files)
        if total_files == 0:
            self.log_message.emit("未找到任何文本文件")
            self.finished_signal.emit(0, 0)
            return
            
        self.log_message.emit(f"找到 {total_files} 个文本文件")
        
        # 开始删除文件
        deleted_count = 0
        for i, file_path in enumerate(txt_files):
            if self.should_stop:
                self.log_message.emit("清理操作已中止")
                break
                
            try:
                # 更新进度
                progress = int((i / total_files) * 100)
                self.update_progress.emit(progress)
                
                # 删除文件
                os.remove(file_path)
                deleted_count += 1
                
                # 记录日志 (只记录前5个和后5个文件，避免日志过长)
                if deleted_count <= 5 or deleted_count > total_files - 5:
                    self.log_message.emit(f"已删除: {file_path}")
                elif deleted_count == 6:
                    self.log_message.emit(f"... 省略 {total_files - 10} 个文件 ...")
                    
            except Exception as e:
                self.log_message.emit(f"删除失败: {file_path}, 错误: {str(e)}")
        
        # 完成清理
        self.update_progress.emit(100)
        self.log_message.emit(f"清理完成，共删除 {deleted_count}/{total_files} 个文本文件")
        self.finished_signal.emit(total_files, deleted_count)


class TextCleanerApp(QMainWindow):
    """文本文件清理应用"""
    def __init__(self):
        super().__init__()
        
        # 设置窗口基本属性
        self.setWindowTitle("文本文件清理工具")
        self.setMinimumSize(600, 400)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题标签
        title_label = QLabel("文本文件清理工具")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        main_layout.addWidget(title_label)
        
        # 说明标签
        description_label = QLabel("此工具用于清理指定目录中的所有.txt文件，包括子文件夹中的文件。")
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)
        
        # 目录选择部分
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("目标目录:"))
        
        self.dir_path_label = QLabel("未选择")
        self.dir_path_label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        self.dir_path_label.setLineWidth(1)
        dir_layout.addWidget(self.dir_path_label, 1)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.browse_btn)
        
        main_layout.addLayout(dir_layout)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始清理")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_cleaning)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_cleaning)
        button_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 日志区域
        log_label = QLabel("处理日志:")
        main_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        
        # 底部状态
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        # 初始化变量
        self.selected_dir = None
        self.cleaner_thread = None
        
        # 添加初始日志
        self.log(f"程序已启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("请选择要清理的目标目录")
    
    def browse_directory(self):
        """浏览并选择目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目标目录")
        if dir_path:
            self.selected_dir = dir_path
            self.dir_path_label.setText(dir_path)
            self.start_btn.setEnabled(True)
            self.log(f"已选择目录: {dir_path}")
    
    def log(self, message):
        """向日志区域添加消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def start_cleaning(self):
        """开始清理文本文件"""
        if not self.selected_dir:
            QMessageBox.warning(self, "警告", "请先选择目标目录")
            return
            
        # 请求确认
        reply = QMessageBox.question(
            self,
            "确认操作",
            f"确定要删除 {self.selected_dir} 目录中的所有.txt文件吗？\n此操作不可撤销!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            self.log("操作已取消")
            return
            
        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 创建并启动清理线程
        self.cleaner_thread = TextCleanerThread(self.selected_dir)
        self.cleaner_thread.update_progress.connect(self.progress_bar.setValue)
        self.cleaner_thread.log_message.connect(self.log)
        self.cleaner_thread.finished_signal.connect(self.on_cleaning_finished)
        
        # 启动线程
        self.cleaner_thread.start()
        self.status_label.setText("正在清理...")
    
    def stop_cleaning(self):
        """停止清理操作"""
        if self.cleaner_thread and self.cleaner_thread.isRunning():
            self.log("正在停止清理操作...")
            self.cleaner_thread.stop()
            self.stop_btn.setEnabled(False)
    
    def on_cleaning_finished(self, total_files, deleted_files):
        """清理完成的回调"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.status_label.setText("清理完成")
        
        # 显示结果对话框
        QMessageBox.information(
            self,
            "清理完成",
            f"清理操作已完成！\n\n总计处理: {total_files} 个文件\n成功删除: {deleted_files} 个文件",
            QMessageBox.Ok
        )


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = TextCleanerApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动错误: {str(e)}")
        QMessageBox.critical(None, "错误", f"程序启动失败: {str(e)}") 