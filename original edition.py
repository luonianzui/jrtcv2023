import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import serial
import serial.tools.list_ports
#sudo chmod 777 /dev/ttyUSB*
#初始化
a = 0x0
b = 0x0
c = 0x0
d = 0x0
i = 0
txbuffer=np.ones(12,np.dtype(np.uint8))
txbuffer[0]=0x11
txbuffer[11]=0x11
#########################################################################
#标定参数导入
# mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# dist = np.array([k1, k2, p1, p2, k3])

##############################################
ser = serial.Serial('/dev/ttyUSB0',115200, timeout=1)  # 将 'TTYUSB0' 替换为你要使用的串口号，115200 是波特率

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model = torch.hub.load('/home/jrt/yolov5-D435i-main', 'custom',
'/home/jrt/yolov5-D435i-main/weights/tge/weights/last.pt',source='local', force_reload=True)
# model_path = 'models/hub/last.pt'
# model = torch.load(model_path)  # 加载模型
model.conf = 0.5
align_to = rs.stream.color 
align = rs.align(align_to)

location = np.zeros(3)  # 添加location变量
location1 = np.zeros(3)

camera_coordinate = np.zeros(3)


# def to_hex_string(num):
#     hex_string = hex(num)[2:]  # 转换为16进制字符串并去掉前缀0x
#     return hex_string.zfill(16)  # 在前面添加零，使其达到16位长度

def get_mid_pos(frame,box,depth_data,randnum):
    global location  # 声明location为全局变量
    # global location1
    global camera_coordinate
   
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

    
    camera_coordinate[0] = int(camera_coordinate[0]) # 将camera_coordinate的值转换为int16类型
    camera_coordinate[1] = int(camera_coordinate[1])
    camera_coordinate[2] = int(camera_coordinate[2])
    # print("camera_coordinate[0]:",camera_coordinate[0])
    # location = location.astype(np.int16)  # 将location的值转换为int16类型
    print("location:",location)
    # print("location1:",location1)

    return np.mean(distance_list),camera_coordinate,location



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
        
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            aligned_frames = align.process(frames)  #获取对齐帧
            aligned_depth_frame = aligned_frames.get_depth_frame()  #获取对齐帧中的depth帧
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

            results = model(color_image)
            boxs= results.pandas().xyxy[0].values
            
            #######################返回三维位置#################
            # x = (box[0] + box[2])//2
            # y = (box[1] + box[3])//2
            # dist = aligned_depth_frame.get_distance(x, y)  #（x, y)点的
            # 真实深度值
            # print("dis: ", dist)
            # output = get_mid_pos.camera_coordinate()
            # print(output)
            # print("get_3D_pos:",location)
            ######################串口发送###############################################################
            # 向串口发送数据
            # data = ','.join(str(value) for value in camera_coordinate) + '\n'  # 将位置向量转换为逗号分隔的字符串
            # ser.write(data.encode())
            #######################接受发送请求######################################################################
            
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
