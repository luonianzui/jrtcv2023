import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import serial
import serial.tools.list_ports
import tkinter as tk
import PIL.Image
#sudo chmod 777 /dev/ttyUSB*
#初始化
bzw = 10 #一个计数器
infobuff = [[0, 0] for i in range(bzw)]
#串口的初始化
a = 0x0
b = 0x0
c = 0x0
d = 0x0
i = 0
x = 0
cnt = 0
txbuffer=np.ones(8,np.dtype(np.uint8))
txbuffer[0]=0x11
txbuffer[7]=0x11
buffer=[]
center_x = 320
center_y = 240
#############hsv的范围####################
hsv = np.empty((0, 0))  #hsv的初始值
hsv_value = np.array([0, 0, 0])
lower = np.array([11, 43, 46])
upper = np.array([25, 255, 255])
##############################################
mid_pos = np.array([0, 0])
box = np.array([0, 0, 0, 0])
ser = serial.Serial('/dev/ttyUSB0',115200, timeout=1)  # 将 'TTYUSB0' 替换为你要使用的串口号，115200 是波特率
model = torch.hub.load('/home/jrt/yolov5-solvepnp', 'custom',
'/home/jrt/yolov5-solvepnp/runs/last.pt',source='local', force_reload=True)
model.conf = 0.5  # 设置置信度阈值
align_to = rs.stream.color 
align = rs.align(align_to)
location = np.zeros(3) 
location1 = np.zeros(3)
camera_coordinate = np.zeros(3)
objects = []

def get_mid_pos(frame,box,depth_data,randnum):
    global location  # 声明location为全局变量
    global camera_coordinate
    global mid_pos
    global hsv_value
    global hsv
    global cnt
    global bzw
    global infobuff
    X = 0
    distance_list = []
    mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
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
    
    #如果camera_coordinate[0]和camera_coordinate[1]的变化量超过阈值，就将infobuff赋值为location
    if (abs(camera_coordinate[0] - infobuff[cnt][0]) > 700 or abs(camera_coordinate[1] - infobuff[cnt][1]) > 700) and camera_coordinate[0] != 0 and camera_coordinate[1] != 0:
            infobuff = [[0, 0] for i in range(bzw)]
            print("ggggggggggggg")
            cnt+=1
            if cnt == 9:
                cnt = 0
            if (x in range(cnt)) and (abs(camera_coordinate[0] - infobuff[x][0]) > 400 or abs(camera_coordinate[1] - infobuff[x][1]) > 400)and camera_coordinate[0] != 0 and camera_coordinate[1] != 0:
                print("cnt:",cnt)      
                infobuff[cnt] = [camera_coordinate[0],camera_coordinate[1]]
                copy = infobuff.copy()
                # 将列表转换为NumPy数组
                infobuff_np = np.array(infobuff)
                copy_np = np.array(copy)
                # 进行矩阵加法
                result_np = infobuff_np + copy_np
                # 将结果转换为列表
                result = result_np.tolist()         
    #转换成hsv
    if hsv is None or hsv.shape[:2] != color_image.shape[:2]:
        hsv = np.zeros_like(color_image)
    #如果取得的mid_pos在图像中，就取hsv值
    if 0 <= mid_pos[0] < hsv.shape[0] and 0 <= mid_pos[1] < hsv.shape[1]:
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        #hsv[(h),(w)]
        hsv_value = hsv[mid_pos[1], mid_pos[0]]
#########################buff###########################
    if (hsv_value[0] >= lower[0] and hsv_value[0] <= upper[0]) and (hsv_value[1] >= lower[1] and hsv_value[1] <= upper[1]) and (hsv_value[2] >= lower[2] and hsv_value[2] <= upper[2]):
        print("location:",location)
    # hsv_value is not orange
    # print("location1:",location1)
    #这部分没写完，今年视觉最后砍了，其实是hsv判断之后返回一个标志值给串口，然后串口根据标志值判断是否开火

    return np.mean(distance_list),camera_coordinate,location,mid_pos,hsv_value



# 更新物体信息，将物体信息存储到objects列表中
def update_object_info(frame, box, depth_data):
    # 获取物体像素中心点位置
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
    # 计算物体在相机坐标系下的位置
    
    # 存储物体信息，包括像素中心点位置和相机坐标系下的位置
    object_info = {'mid_pos': mid_pos, 'location': camera_coordinate}
    # 将物体信息添加到objects列表中
    #如果mid__pos的变化x和y的值大于40，就将物体信息添加到objects列表中
    if len(objects) == 0 or abs(objects[-1]['mid_pos'][0] - mid_pos[0]) > 40 or abs(objects[-1]['mid_pos'][1] - mid_pos[1]) > 40:
        objects.append(object_info)
    # 对objects列表按照物体像素中心点从小到大排序
    objects.sort(key=lambda obj: obj['mid_pos'][0])
    # 将排序后的物体信息依次存储到txbuffer数组中
    for i, obj in enumerate(objects):
        buffer[i] = obj['location']
        print("buffer:",buffer[i])

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

def dectshow(org_img, boxs,depth_data):
    global center_x, center_y
    img = org_img.copy()
    random_num = random.randint(50,60)
    cv2.line(img, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 0), 2)
    cv2.line(img, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 2)
    #在左上角写fps：random_num，写着玩的，可以删掉
    cv2.putText(img, "fps:" + str(random_num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
            aligned_frames = align.process(frames)  #获取对齐帧
            aligned_depth_frame = aligned_frames.get_depth_frame()  #获取对齐帧中的depth帧
            if not depth_frame1 or not color_frame1:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame1.get_data())

            color_image = np.asanyarray(color_frame1.get_data())
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            RGB_image = color_image[..., ::-1]
            results = model(RGB_image)
            # results = model(color_image)

            boxs= results.pandas().xyxy[0].values
            #串口部分
            if ser.in_waiting:
                    #读取一行数据，判断是否为“aaabacad” ，是才发送，不是就丢弃，第一年写，搞得不行，暑假学习了crc校验，但是还没写               
                    # data = ser.read(8).hex()
                    # if data[i] == 0x61 and data[i+1] == 0x61 :
                    #     if data[i+2] == 0x61 and data[i+3] == 0x62 :
                    #         if data[i+4] == 0x61 and data[i+5] == 0x63 :
                    #             if data[i+6] == 0x61 and data[i+7] == 0x64 :
                                    #发送
                                    # transform = np.array(camera_coordinate[0], dtype=np.float64)
                                    # binary_arr = np.array([np.binary_repr(num, width=16) for num in transform], dtype=np.uint16)
                                    # binary_num = (num & 0xFFFF)
                                    #想发什么轴的值都能改，这里只是举例子，但是这里的z其实是深度值，要经过运算才能得到z轴的值
                                    ################z轴##############################
                                    if camera_coordinate[2] < 0:
                                        camera_coordinate[2] = -camera_coordinate[2]
                                        #规定符号位
                                        txbuffer[1] = 0x01
                                    else:
                                        txbuffer[1] = 0x00
                                        # camera_coordinate[0] = camera_coordinate[0] | 0x8000
                                    zbinary_arr = (camera_coordinate[2] & 0xFFFF)
                                    zhigh_byte = (zbinary_arr >> 8) & 0xff
                                    zlow_byte = zbinary_arr & 0xff
                                    txbuffer[2]=zhigh_byte
                                    txbuffer[3]=zlow_byte
                                    print(zbinary_arr)
                                    print(txbuffer[1])
                                    print(zhigh_byte)
                                    print(txbuffer[2])
                                    print(zlow_byte)
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

                                    ################class分类器##############################
                                    # class = class.astype(np.int)
                                    # class = boxs[-1]
                                    # txbuffer[10]=class
                                    # print(class)
                                    #########################################################
                                    
                                    # data = ','.join(str(value) for value in hex_arr ) + '\n'  # 将位置向量转换为逗号分隔的字符串
                                    ser.write(txbuffer)

           
            dectshow(color_image, boxs, depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            
            cv2.imshow('RealSense', images)
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
