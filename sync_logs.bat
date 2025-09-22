@echo off
echo 正在从远程服务器同步日志文件...

REM 请修改以下变量为您的服务器信息
set SERVER_USER=your_username
set SERVER_IP=your_server_ip
set REMOTE_PATH=/path/to/LoraRetriever/logs/
set LOCAL_PATH=.\logs\

REM 创建本地logs目录
if not exist "%LOCAL_PATH%" mkdir "%LOCAL_PATH%"

REM 使用scp下载日志文件
echo 下载日志文件...
scp -r %SERVER_USER%@%SERVER_IP%:%REMOTE_PATH%* %LOCAL_PATH%

echo 同步完成！
echo 日志文件已保存到: %LOCAL_PATH%
pause