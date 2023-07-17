import pyrealsense2 as rs
import numpy as np
import cv2
# import scipy.misc
import time
import sys

def get_image():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    # Start streaming
    pipeline.start(config)

    #获取图像，realsense刚启动的时候图像会有一些失真，我们保存第100帧图片。
    for i in range(100):
        data = pipeline.wait_for_frames()
        depth = data.get_depth_frame()
        color = data.get_color_frame()
        ir_left = data.get_infrared_frame(1)
        ir_right = data.get_infrared_frame(2)
    #获取内参
    dprofile = depth.get_profile()
    cprofile = color.get_profile()
    lprofile = ir_left.get_profile()
    rprofile = ir_right.get_profile()

    cvsprofile = rs.video_stream_profile(cprofile)
    dvsprofile = rs.video_stream_profile(dprofile)
    lvsprofile = rs.video_stream_profile(lprofile)
    rvsprofile = rs.video_stream_profile(rprofile)

    color_intrin=cvsprofile.get_intrinsics()
    print(color_intrin)
    depth_intrin=dvsprofile.get_intrinsics()
    print(depth_intrin)
    ir_left_intrin = lvsprofile.get_intrinsics()
    print(ir_left_intrin)
    ir_right_intrin = rvsprofile.get_intrinsics()
    print(ir_right_intrin)

    #外参
    # extrin = dprofile.get_extrinsics_to(cprofile)
    # print(extrin)
    extrin = lprofile.get_extrinsics_to(rprofile)
    print(extrin)

    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())
    ir_left_image = np.asanyarray(ir_left.get_data())
    ir_right_image = np.asanyarray(ir_right.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Show images
    images0 = np.hstack((ir_left_image, ir_right_image))
    cv2.imshow('images0', images0)
    # cv2.imshow('ir_left_image', ir_left_image)
    # cv2.imshow('ir_right_image', ir_right_image)
    cv2.waitKey(100)

    t = time.time()
    tname = str(t)[5:10]

   # cv2.imwrite('ir_left_image' + str(tname) + '.png', ir_left_image)
   # cv2.imwrite('ir_right_image' + str(tname) + '.png', ir_right_image)

    cv2.imwrite('color' + str(tname) + '.png', color_image)
    #cv2.imwrite('depth' + str(tname) + '.png', depth_image)
    #cv2.imwrite('depth_colorMAP' + str(tname) + '.png', depth_colormap)
    # scipy.misc.imsave('outfile1.png', depth_image)
    # scipy.misc.imsave('outfile2.png', color_image)
if __name__ == "__main__":
    while(1):
            char = sys.stdin.read(1)
            if char == 'q':
                x = 1
         
            if x== 1:
                get_image()
                x=0
                
