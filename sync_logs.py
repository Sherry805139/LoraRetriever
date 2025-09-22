#!/usr/bin/env python3
"""
自动同步远程服务器日志文件到本地
"""

import subprocess
import os
import time
from datetime import datetime

# 服务器配置 - 请修改为您的服务器信息
SERVER_CONFIG = {
    'user': 'your_username',           # 您的用户名
    'host': 'your_server_ip',          # 服务器IP
    'remote_path': '/path/to/LoraRetriever/logs/',  # 远程logs路径
    'local_path': './logs/',           # 本地logs路径
}

def sync_logs():
    """同步日志文件"""
    print(f"[{datetime.now()}] 开始同步日志文件...")
    
    # 创建本地logs目录
    os.makedirs(SERVER_CONFIG['local_path'], exist_ok=True)
    
    # 构建rsync命令
    cmd = [
        'rsync', '-avz', '--progress',
        f"{SERVER_CONFIG['user']}@{SERVER_CONFIG['host']}:{SERVER_CONFIG['remote_path']}",
        SERVER_CONFIG['local_path']
    ]
    
    try:
        # 执行同步
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 同步成功！")
            print(f"输出: {result.stdout}")
        else:
            print("❌ 同步失败！")
            print(f"错误: {result.stderr}")
            
    except FileNotFoundError:
        print("❌ 未找到rsync命令，请尝试使用scp")
        # 备用scp命令
        cmd_scp = [
            'scp', '-r',
            f"{SERVER_CONFIG['user']}@{SERVER_CONFIG['host']}:{SERVER_CONFIG['remote_path']}*",
            SERVER_CONFIG['local_path']
        ]
        try:
            result = subprocess.run(cmd_scp, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ SCP同步成功！")
            else:
                print(f"❌ SCP同步失败: {result.stderr}")
        except Exception as e:
            print(f"❌ 同步失败: {e}")
    
    # 列出本地日志文件
    print("\n本地日志文件:")
    try:
        for file in os.listdir(SERVER_CONFIG['local_path']):
            if file.endswith('.log') or file.endswith('.txt'):
                file_path = os.path.join(SERVER_CONFIG['local_path'], file)
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size} bytes)")
    except Exception as e:
        print(f"无法列出文件: {e}")

def monitor_logs():
    """持续监控并同步日志"""
    print("开始监控模式，每30秒同步一次...")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            sync_logs()
            print(f"\n等待30秒后下次同步...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_logs()
    else:
        sync_logs()
        
    print(f"\n使用方法:")
    print(f"python sync_logs.py          # 单次同步")
    print(f"python sync_logs.py monitor  # 持续监控同步")