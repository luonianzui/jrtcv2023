江苏第二师范学院2023cv-企鹅号2071676670（菜狗）,欢迎交流
使用神经网络和hsv结合的方法

1.获取数据集
get_photo.py 是相机拍照取帧的脚本 ,获取数据集使用的
运行文件按q，一下取一个帧，但是加入了cv2.waitKey(100)，所以有一定的延迟，根据自己的需求修改
2.炼丹
https://blog.csdn.net/qq_64036218/article/details/127859958?spm=1001.2014.3001.5502
具体参考我自己记录的博客
3.模型使用
在mian里model = torch.hub.load('/home/jrt/yolov5-solvepnp', 'custom',
'/home/jrt/yolov5-solvepnp/runs/last.pt',source='local', force_reload=True)
model.conf = 0.5  # 设置置信度阈值，调用了本地的炼丹模型

#代码的完整仓库在https://github.com/luonianzui/yolov5-solvepnp.git

#track_lose.py是写track失败的文件，时间不够了，明年使用cpp重构的时候顺手写了

#calibration.py这个文件可获取相机的内参和外参，标定脚本




####################################################################
基本上是从original edition.py修改过来的
本项目基于yolov5(https://github.com/ultralytics/yolov5)
将D435深度相机和yolov5结合到一起，在识别物体的同时，还能测到物体相对与相机的距离
硬件准备：
D435i是一个搭载IMU（惯性测量单元，采用的博世BMI055）的深度相机，D435i的2000万像素RGB摄像头和3D传感器可以30帧/秒的速度提供分辨率高达1280 × 720，或者以90帧/秒的速度提供848 × 480的较低分辨率。该摄像头为全局快门，可以处理快速移动物体，室内室外皆可操作。深度距离在0.1 m~10 m之间
计算机 win10 or ubuntu 最好有nvidia显卡
软件准备：
使用pip安装所需的包，进入本工程目录下
pip install -r requirements.txt
pip install pyrealsense2
# 程序运行
命令行cd 进入工程文件夹下
python3 main_debug.py
注意： 第一次运行程序程序会从云端下载yolov5的pt文件，大约140MB+ 
#####################################################################


