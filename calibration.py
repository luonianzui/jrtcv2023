# -*- coding=utf-8 -*-
# name: nan chen
# date: 2021/5/18 10:50

import cv2
import numpy as np
import glob

np.set_printoptions(suppress=True)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

w = 8 
h = 11

objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

images = glob.glob('*.jpg')
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
   
    if ret == True:
        # 角点精确检测
   
        i += 1
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.imwrite('p2/h' + str(i) + '.jpg', img)
        cv2.waitKey(10)
cv2.destroyAllWindows()
# 标定、去畸变

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# mtx：内参数矩阵
# dist：畸变系数
# rvecs：旋转向量 （外参数）
# tvecs ：平移向量 （外参数）
print(("ret:"), ret)
print(("mtx:\n"), mtx)  
print(("dist:\n"), dist) 
print(("rvecs:\n"), rvecs) 
print(("tvecs:\n"), tvecs)  
# 去畸变
img2 = cv2.imread('p2/1.jpg')
h, w = img2.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数

dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

cv2.imwrite('p2/calibresult.jpg', dst)


total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print(("total error: "), total_error / len(objpoints))

