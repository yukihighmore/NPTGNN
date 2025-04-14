
import subprocess
import time

import winsound
import pyttsx3
from playsound import playsound
import itchat

class CommandException(Exception):
    pass


def run_cmd(command, log_path=None, times=0):
    # 执行命令
    f = open(log_path, 'a')
    f.writelines(str(times) + '\n')
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 逐行读取输出，并实时输出到控制台

    for line in iter(p.stdout.readline, b''):
        str_line = line.decode("utf-8")
        print(str_line, end='')
        if 'ETA' not in str_line and 'Epoch' not in str_line:
            f.writelines(str_line)

    # 逐行读取错误输出，并实时输出到控制台
    for line in iter(p.stderr.readline, b''):
        print(line.decode("utf-8"), end='')

    # 等待子进程结束
    p.communicate()
    f.close()

if __name__ == '__main__':

    from sendwx import send  # 我之前保存的是sendwx.py

    send('开始了，开始了')

    engine = pyttsx3.init()  # 创建engine并初始化
    rate = engine.getProperty('rate')  # 获取当前语速的详细信息
    print(rate)  # 打印当前语速
    engine.setProperty('rate', 125)
    engine.say("开始了，开始了")
    engine.runAndWait()  # 等待语音播报完毕

    python_path = 'C:/Users/HP/anaconda3/envs/Metro/python.exe'


    target_py = 'C:/YukiCode/Metro/run_SH.py'
    cmd = python_path + ' ' + target_py
    log_path = 'C:/YukiCode/Metro/run_SH.txt'
    for i in range(10):
        run_cmd(cmd, log_path, i)
        time.sleep(30)



    # output:  /Users/myproject/mydemo/demo
    engineEnd = pyttsx3.init()  # 创建engine并初始化
    rate = engineEnd.getProperty('rate')  # 获取当前语速的详细信息
    print(rate)  # 打印当前语速
    engineEnd.setProperty('rate', 100)
    engineEnd.say("结束了，结束了，结束了，结束了，结束了")
    engineEnd.runAndWait()  # 等待语音播报完毕

    send('结束了，结束了')

