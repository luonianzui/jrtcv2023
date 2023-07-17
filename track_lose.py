import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import serial
import serial.tools.list_ports
from shapely.geometry import box as shapely_box
from scipy.optimize import linear_sum_assignment
#sudo chmod 777 /dev/ttyUSB*
#初始化
a = 0x0
b = 0x0
c = 0x0
d = 0x0
i = 0
iou = 0.000
txbuffer=np.ones(12,np.dtype(np.uint8))
txbuffer[0]=0x11
txbuffer[11]=0x11
box1 = np.array([0, 0, 0, 0])
box2 = np.array([0, 0, 0, 0])
dt = 1 / 60
#############hsv的范围####################
hsv = np.empty((0, 0))  #hsv的初始值
hsv_value = np.array([0, 0, 0])
lower = np.array([11, 43, 46])
upper = np.array([25, 255, 255])
mid_pos = np.array([0, 0])
boxs = np.array([0, 0, 0, 0, 0 ,0 ,0])
depth_image1 = np.zeros((480, 640), np.uint8)
depth_image2 = np.zeros((480, 640), np.uint8)
##############################################
ser = serial.Serial('/dev/ttyUSB0',115200, timeout=1)  # 将 'TTYUSB0' 替换为你要使用的串口号，115200 是波特率

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model = torch.hub.load('/home/jrt/yolov5-solvepnp', 'custom',
'/home/jrt/yolov5-solvepnp/runs/last.pt',source='local', force_reload=True)
# model_path = 'models/hub/last.pt'
# model = torch.load(model_path)  # 加载模型
model.conf = 0.5  # 设置置信度阈值
align_to = rs.stream.color 
align = rs.align(align_to)

location = np.zeros(3)  # 添加location变量
location1 = np.zeros(3)

camera_coordinate = np.zeros(3)


# def to_hex_string(num):
#     hex_string = hex(num)[2:]  # 转换为16进制字符串并去掉前缀0x
#     return hex_string.zfill(16)  # 在前面添加零，使其达到16位长度
##############################################dabian########################################################

############################################################################################################

def get_mid_pos(frame,box,depth_data,randnum):
    global location  # 声明location为全局变量
    # global location1
    global camera_coordinate
    global mid_pos
    global hsv_value
    global hsv
    # global mid_pos
   
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        # dis = aligned_depth_frame.get_distance(mid_pos) 
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    #（x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
    camera_coordinate = (rs.rs2_deproject_pixel_to_point(depth_intrin, [int(mid_pos[1] + bias), int(mid_pos[0] + bias)], dist))  
    # print("66:",camera_coordinate,box[-1])
    location = np.array([camera_coordinate[0],camera_coordinate[1],camera_coordinate[2]])  # 将location赋值为相机坐标系下的xyz值
    # location1 = location
    mid_pos[0] = int(mid_pos[0])
    mid_pos[1] = int(mid_pos[1])
    camera_coordinate[0] = int(camera_coordinate[0]) # 将camera_coordinate的值转换为int16类型
    camera_coordinate[1] = int(camera_coordinate[1])
    camera_coordinate[2] = int(camera_coordinate[2])
    #转换成hsv
    if hsv is None or hsv.shape[:2] != color_image.shape[:2]:
        hsv = np.zeros_like(color_image)
    #如果取得的mid_pos在图像中，就取hsv值
    #确保mid_pos的值在hsv图像的大小范围内
    # if mid_pos[0] >= 0 and mid_pos[0] < hsv.shape[1] and mid_pos[1] >= 0 and mid_pos[1] < hsv.shape[0]:
    #     # hsv_value = hsv[mid_pos[1], mid_pos[0]]
    #     # mask = cv2.inRange(hsv, lower, upper)
    #     hsv_value = hsv[mid_pos[0], mid_pos[1]]
    if 0 <= mid_pos[0] < hsv.shape[0] and 0 <= mid_pos[1] < hsv.shape[1]:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        #hsv[(h),(w)]
        hsv_value = hsv[mid_pos[1], mid_pos[0]]
    
    #如果hsv_value在lower和upper之间，就发送数据
    # # print("camera_coordinate[0]:",camera_coordinate[0])
    # # location = location.astype(np.int16)  # 将location的值转换为int16类型
    if (hsv_value[0] >= lower[0] and hsv_value[0] <= upper[0]) and (hsv_value[1] >= lower[1] and hsv_value[1] <= upper[1]) and (hsv_value[2] >= lower[2] and hsv_value[2] <= upper[2]):
    # if is_orange(hsv_value):
    # hsv_value is orange
        print("location:",location)
    
    # hsv_value is not orange
    # print("location1:",location1)

    return np.mean(distance_list),camera_coordinate,location,mid_pos,hsv_value

def is_orange(hsv_value):
    # Check if hsv_value is a valid numpy array
    if  hsv_value.shape != (1,1,3):
        return False
        
    # Define lower and upper bounds for orange color
    lower_orange = np.array([0, 127, 127])
    upper_orange = np.array([22, 255, 255])
    lower_orange2 = np.array([160, 127, 127])
    upper_orange2 = np.array([180, 255, 255])

    # Check if hsv_value is within orange color range
    mask_orange = cv2.inRange(hsv_value, lower_orange, upper_orange) | cv2.inRange(hsv_value, lower_orange2, upper_orange2)
    count = cv2.countNonZero(mask_orange)
    return count > 0


def get_iou(box1, box2):
    # 取出 box1 和 box2 的左上角和右下角坐标
    x1, y1, x2, y2 = box1[0:4]
    u1, v1, u2, v2 = box2[0:4]

    # 计算 box1 和 box2 的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (u2 - u1) * (v2 - v1)

    # 计算 box1 和 box2 的交集部分
    overlap_x1 = max(x1, u1)
    overlap_y1 = max(y1, v1)
    overlap_x2 = min(x2, u2)
    overlap_y2 = min(y2, v2)
    overlap_w = max(0, overlap_x2 - overlap_x1)
    overlap_h = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_w * overlap_h

    # 计算 box1 和 box2 的并集部分
    union_area = area1 + area2 - overlap_area

    # 计算交并比并返回
    iou = overlap_area / union_area if union_area > 0 else 0
    return iou


# def get_deep_compare(box1,box2,depth_image1,depth_image2):
#     mid_pos1 = [(box1[0] + box1[2])//2, (box1[1] + box1[3])//2] #确定索引深度的中心像素位置
#     mid_pos2 = [(box2[0] + box2[2])//2, (box2[1] + box2[3])//2]
#     mid_pos1 = (int(mid_pos1[0]), int(mid_pos1[1]))
#     mid_pos2 = (int(mid_pos2[0]), int(mid_pos2[1]))
#     # min_val1 = min(abs(box1[2] - box1[0]), abs(box1[3] - box1[1])) #确定深度搜索范围
#     # min_val2 = min(abs(box2[2] - box2[0]), abs(box2[3] - box2[1]))
#     deep_compare = abs(depth_image1[mid_pos1[1], mid_pos1[0]] - depth_image2[mid_pos2[1], mid_pos2[0]])
#     return deep_compare


def get_deep_compare(box1, box2, depth_image1, depth_image2):
    mid_pos1 = [(box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2]
    mid_pos2 = [(box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2]
    mid_pos1 = [np.clip(mid_pos1[0], 0, depth_image1.shape[1] - 1), np.clip(mid_pos1[1], 0, depth_image1.shape[0] - 1)]
    mid_pos2 = [np.clip(mid_pos2[0], 0, depth_image2.shape[1] - 1), np.clip(mid_pos2[1], 0, depth_image2.shape[0] - 1)]

    # 确定深度搜索范围
    min_val1 = (box1[2] - box1[0]) // 2
    min_val2 = (box2[2] - box2[0]) // 2

    # 计算两个框内所有像素的深度值的中位数，并计算它们之间的差异
    depth_values1 = depth_image1[
        int(mid_pos1[1] - min_val1 // 2) : int(mid_pos1[1] + min_val1 // 2),
        int(mid_pos1[0] - min_val1 // 2) : int(mid_pos1[0] + min_val1 // 2),
    ]
    depth_values2 = depth_image2[
        int(mid_pos2[1] - min_val2 // 2) : int(mid_pos2[1] + min_val2 // 2),
        int(mid_pos2[0] - min_val2 // 2) : int(mid_pos2[0] + min_val2 // 2),
    ]
    depth_median1 = np.median(depth_values1)
    depth_median2 = np.median(depth_values2)
    deep_compare = abs(depth_median1 - depth_median2)

    return deep_compare

def track_decision(box1, box2, depth_image1, depth_image2):
    iou = get_iou(box1, box2)
    deep_compare = get_deep_compare(box1, box2, depth_image1, depth_image2)

    if iou > 0.5 and deep_compare < 100 and box1[-1] == box2[-1]:
        # print('track success')
        return True
    else:
        # print('track failed')
        return False


class Tracker:
    def __init__(self, init_pos, dt):
        self.pos = init_pos # 初始位置
        self.vel = np.array([0.0, 0.0, 0.0]) # 初始速度
        self.dt = dt # 时间步长
        self.KF = self._init_kalman_filter(init_pos)# 初始化卡尔曼滤波器
        self.state = 'DETECTING' # 跟踪器状态
        self.max_lost = 60 # 最大丢失帧数

    def _init_kalman_filter(self, init_pos):
      
        # 定义卡尔曼滤波器
        KF = cv2.KalmanFilter(6, 3)
        # 状态转移矩阵
        KF.transitionMatrix = np.array([[1, 0, 0, self.dt, 0, 0],
                                        [0, 1, 0, 0, self.dt, 0],
                                        [0, 0, 1, 0, 0, self.dt],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        # 测量矩阵
        KF.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0]], dtype=np.float32)
        # 过程噪声协方差矩阵
        KF.processNoiseCov = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
        # 测量噪声协方差矩阵
        KF.measurementNoiseCov = np.diag([1e-1, 1e-1, 1e-1])
        # 初始状态
        KF.statePost = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0], dtype=np.float32)
        return KF

    def update(self, detections):
    # 用卡尔曼滤波器预测目标位置
        self.KF.predict()
        pred_pos = self.KF.statePre[:3]

        if detections is None or len(detections) == 0:
            
            if self.lost_count < self.max_lost and track_decision == False:
                # 目标丢失帧数小于最大允许丢失帧数，继续等待目标出现
                self.lost_count += 1
                self.state = 'TEMP_LOST'
                
                return None
            else:
               
                self.state = 'LOST'
                print('lost')
                return None

        # 计算当前帧中所有目标位置与预测位置的距离
        dist = np.linalg.norm(detections - pred_pos, axis=1)

        # 找到位置差最小的目标，作为最佳匹配项
        idx = np.argmin(dist)
        min_dist = dist[idx]
        if min_dist > 50:
            # 位置偏差过大，判断为目标丢失
            self.state = 'LOST'
            return None

        # 更新卡尔曼滤波器状态
        self.KF.correct(detections[idx])
        self.pos = self.KF.statePost[:3]
        self.vel = self.KF.statePost[3:6]

        if self.state == 'DETECTING':
            if track_decision == True and min_dist >= 50:

            # 检测到目标，但还没有足够的信息进行跟踪，需要更多帧的检测结果
            
                self.state = 'TRACKING'
                print('track success')
        else:
            
            self.state = 'TEMP_LOST'

        # 重置目标丢失计数器
        self.lost_count = 0

        return self.pos


def dectshow(org_img, boxs,depth_data):
    img = org_img.copy()
    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        dist = get_mid_pos(org_img, box, depth_data, 24)
        cv2.putText(img, box[-1] + str(dist)[:6] + 'mm',
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('dec_img', img)

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    ##########################################################################################################
# -----------------------------------------------
 #打印出来get_3D_pos的值
    # print("get_3D_pos:",get_3D_pos(frame,box,depth_data,24))
    # print("66:",camera_coordinate,box[-1])
    ##########################################################################################################
    try:
        while True:
        
            # 获取前一帧
            frames = pipeline.wait_for_frames()
            depth_frame1 = frames.get_depth_frame()
            color_frame1 = frames.get_color_frame()
            aligned_frames1 = align.process(frames)  #获取对齐帧
            aligned_depth_frame1 = aligned_frames1.get_depth_frame()  #获取对齐帧中的depth帧
            if not depth_frame1 or not color_frame1:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame1.get_data())
            color_image = np.asanyarray(color_frame1.get_data())
            depth_intrin = aligned_depth_frame1.profile.as_video_stream_profile().intrinsics
            RGB_image = color_image[..., ::-1]
            results1 = model(RGB_image)
            
            # results = model(color_image)

            boxs= results1.pandas().xyxy[0].values
            #从将boxs的格式从tensor转换为numpy
            if len(boxs) == 0:
                print("ggggggggggg")
            else:    
                boxs=boxs.tolist()
                box1=boxs[0]
                x1, y1, x2, y2, conf, cls, cls_conf = map(float, boxs[0])
                cls_conf = float(cls_conf)
                depth_image1 = np.copy(depth_image)
                box1 = [x1, y1, x2, y2, conf, cls, cls_conf]
            


            ######################################################################################
            #如果hsv_value在lower和upper之间，就发送数据
            # if (hsv_value[0] >= lower[0] and hsv_value[0] <= upper[0]) and (hsv_value[1] >= lower[1] and hsv_value[1] <= upper[1]) and (hsv_value[2] >= lower[2] and hsv_value[2] <= upper[2]):
            if ser.in_waiting:
                    # data = ser.read(8).hex()
                    # if data[i] == 0x61 and data[i+1] == 0x61 :
                    #     if data[i+2] == 0x61 and data[i+3] == 0x62 :
                    #         if data[i+4] == 0x61 and data[i+5] == 0x63 :
                    #             if data[i+6] == 0x61 and data[i+7] == 0x64 :
                                    #发送
                                    # transform = np.array(camera_coordinate[0], dtype=np.float64)
                                    # binary_arr = np.array([np.binary_repr(num, width=16) for num in transform], dtype=np.uint16)
                                    # binary_num = (num & 0xFFFF)
                                    ################y轴##############################
                                    if camera_coordinate[0] < 0:
                                        camera_coordinate[0] = -camera_coordinate[0]
                                        #规定符号位
                                        txbuffer[1] = 0x01
                                    else:
                                        txbuffer[1] = 0x00
                                        # camera_coordinate[0] = camera_coordinate[0] | 0x8000
                                    ybinary_arr = (camera_coordinate[0] & 0xFFFF)
                                    yhigh_byte = (ybinary_arr >> 8) & 0xff
                                    ylow_byte = ybinary_arr & 0xff
                                    txbuffer[2]=yhigh_byte
                                    txbuffer[3]=ylow_byte
                                    print(ybinary_arr)
                                    print(txbuffer[1])
                                    print(yhigh_byte)
                                    print(txbuffer[2])
                                    print(ylow_byte)
                                    print(txbuffer[3])
                                    ################x轴##############################           
                                    if camera_coordinate[1] < 0:
                                        camera_coordinate[1] = -camera_coordinate[1]
                                        #规定符号位
                                        txbuffer[4] = 0x01
                                    else:
                                        txbuffer[4] = 0x00
                                        # camera_coordinate[0] = camera_coordinate[0] | 0x8000
                                    xbinary_arr = (camera_coordinate[1] & 0xFFFF)
                                    xhigh_byte = (xbinary_arr >> 8) & 0xff
                                    xlow_byte = xbinary_arr & 0xff
                                    txbuffer[5]=xhigh_byte
                                    txbuffer[6]=xlow_byte
                                    print(xbinary_arr)
                                    print(txbuffer[4])
                                    print(xhigh_byte)
                                    print(txbuffer[5])
                                    print(xlow_byte)
                                    print(txbuffer[6])
                                    ################z轴##############################
                                    if camera_coordinate[2] < 0:
                                        camera_coordinate[2] = -camera_coordinate[2]
                                        #规定符号位
                                        txbuffer[7] = 0x01
                                    else:
                                        txbuffer[7] = 0x00
                                        # camera_coordinate[0] = camera_coordinate[0] | 0x8000
                                    zbinary_arr = (camera_coordinate[2] & 0xFFFF)
                                    zhigh_byte = (zbinary_arr >> 8) & 0xff
                                    zlow_byte = zbinary_arr & 0xff
                                    txbuffer[8]=zhigh_byte
                                    txbuffer[9]=zlow_byte
                                    print(zbinary_arr)
                                    print(txbuffer[7])
                                    print(zhigh_byte)
                                    print(txbuffer[8])
                                    print(zlow_byte)
                                    print(txbuffer[9])
                                    ################class分类器##############################
                                    # class = class.astype(np.int)
                                    # class = boxs[-1]
                                    # txbuffer[10]=class
                                    # print(class)
                                    #########################################################
                                    
                                    # data = ','.join(str(value) for value in hex_arr ) + '\n'  # 将位置向量转换为逗号分隔的字符串
                                    ser.write(txbuffer)





            # data = ser.readline()

            # print('Received:', data)
            
            #################################################################
            
            #boxs = np.load('temp.npy',allow_pickle=True)
            dectshow(color_image, boxs, depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            
            cv2.imshow('RealSense', images)
            # if box1 is not None and box2 is not None:
            
            # box1 = box1[0:4]
            # box2 = box2[0:4]
            # #如果是第一次运行，初始化
            # tracker = Tracker(location, dt)
            # if tracker is None:
            #     Tracker.__init__(location,dt)
            #     Tracker._init_kalman_filter(location)
            #     print("init")
            #如果不是第一次运行，更新
            # else:
                # Tracker.update(location)

            # Tracker.update(box1,box2)
            #这部分祭拉
            # if box1 is not None and box2 is not None:
            #     track_decision(box1,box2,depth_image1,depth_image2)
                # get_iou(box1,box2)
                # get_deep_compare(box1,box2,depth_image1,depth_image2)
                # print("iou:",get_iou(box1,box2))
                # print("deep_compare:",get_deep_compare(box1,box2,depth_image1,depth_image2))
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # 关闭串口连接
        ser.close()
        # Stop streaming
        pipeline.stop()
