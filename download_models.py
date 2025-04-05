#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import urllib.request
import shutil
from pathlib import Path
import time

# 模型文件URLs
YOLOV3_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLOV3_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"

# 保存路径
MODELS_DIR = "models"


def download_with_progress(url, dest_path):
    """
    带进度显示的文件下载
    """
    if os.path.exists(dest_path):
        print(f"文件已存在: {dest_path}")
        return True
        
    print(f"正在下载: {url}")
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.info().get('Content-Length', 0))
            
            # 创建临时下载文件
            temp_path = dest_path + ".download"
            
            # 如果存在未完成的下载，删除它
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            
            with open(temp_path, 'wb') as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                        
                    downloaded += len(chunk)
                    out_file.write(chunk)
                    
                    # 计算并显示进度
                    percent = downloaded / file_size * 100 if file_size > 0 else 0
                    elapsed_time = time.time() - start_time
                    speed = downloaded / elapsed_time / 1024 / 1024 if elapsed_time > 0 else 0
                    
                    sys.stdout.write(f"\r下载进度: {percent:.1f}% ({downloaded/1024/1024:.1f} MB / {file_size/1024/1024:.1f} MB) "
                                    f"速度: {speed:.2f} MB/s")
                    sys.stdout.flush()
            
            print()  # 换行
            
            # 下载完成，重命名文件
            os.rename(temp_path, dest_path)
            return True
            
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        # 清理未完成的下载
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def main():
    # 创建models目录
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # 下载配置文件
    cfg_path = os.path.join(MODELS_DIR, "yolov3.cfg")
    if not download_with_progress(YOLOV3_CFG_URL, cfg_path):
        print("无法下载YOLOv3配置文件")
        return False
        
    # 下载权重文件（较大文件）
    weights_path = os.path.join(MODELS_DIR, "yolov3.weights")
    if not download_with_progress(YOLOV3_WEIGHTS_URL, weights_path):
        print("无法下载YOLOv3权重文件")
        return False
        
    print("\n模型文件下载完成！")
    print(f"YOLOv3配置文件: {os.path.abspath(cfg_path)}")
    print(f"YOLOv3权重文件: {os.path.abspath(weights_path)}")
    
    return True


if __name__ == "__main__":
    print("=== YOLOv3模型下载工具 ===")
    main() 