# -*- coding:utf-8 -*-
__author__ = 'Microcosm'

import cv2
import numpy as np
import glob
import sys

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
nrows = 4
ncols = 6
objp = np.zeros((nrows*ncols,3), np.float32)
objp[:,:2] = np.mgrid[0:ncols,0:nrows].T.reshape(-1,2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
objp *= 0.052
# print objp
dir = sys.argv[1]
obj_points = []    # 存储3D点
img_points = []    # 存储2D点

images = glob.glob(dir + "/*.png")
# images = glob.glob(dir + "/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    # print fname
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]

    #cv2.imshow('img', gray)
    #cv2.waitKey(0)

    ret, corners = cv2.findChessboardCorners(gray, (ncols,nrows), None)

    if ret:
        print fname
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)  # 在原角点的基础上寻找亚像素角点
        # print corners2
        if corners2.any():
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (ncols,nrows), corners, ret)   # 记住，OpenCV的绘制函数一般无返回值
        cv2.imshow('img', img)
        cv2.waitKey(20)

print len(img_points)
cv2.destroyAllWindows()

# 标定
#flags = cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
flags = cv2.CALIB_FIX_K3
# flags = cv2.CALIB_FIX_K1
# flags |= cv2.CALIB_FIX_K2
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5
# flags |= cv2.CALIB_FIX_K6
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray.shape[::-1], None, None, flags= flags)

print "ret:",ret
print "mtx:\n",mtx        # 内参数矩阵
print "dist:\n",dist      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#print "rvecs:\n",rvecs    # 旋转向量  # 外参数
#print "tvecs:\n",tvecs    # 平移向量  # 外参数

print("-----------------------------------------------------")
# 畸变校正
# img = cv2.imread(images[12])
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# print newcameramtx
# print("------------------使用undistort函数-------------------")
# dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
# x,y,w,h = roi
# dst1 = dst[y:y+h,x:x+w]
# cv2.imwrite(dir+'/calibresult11.jpg', dst1)
# print "方法一:dst的大小为:", dst1.shape

# undistort方法二
# print("-------------------使用重映射的方式-----------------------")
# mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)  # 获取映射方程
# #dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)      # 重映射
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_CUBIC)        # 重映射后，图像变小了
# x,y,w,h = roi
# dst2 = dst[y:y+h,x:x+w]
# cv2.imwrite(dir+'/calibresult11_2.jpg', dst2)
# print "方法二:dst的大小为:", dst2.shape        # 图像比方法一的小

# print("-------------------计算反向投影误差-----------------------")
# tot_error = 0
# for i in xrange(len(obj_points)):
#     img_points2, _ = cv2.projectPoints(obj_points[i],rvecs[i],tvecs[i],mtx,dist)
#     error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
#     tot_error += error
#
# mean_error = tot_error/len(obj_points)
# print "total error: ", tot_error
# print "mean error: ", mean_error

# import cv2
# import numpy as np
# import glob
# import sys
#
# dir = sys.argv[1]
# mtx = np.array([[ 554.07696251, 0., 494.28592613],
#  [   0., 550.33466473, 263.92806352],
#  [   0., 0.,1.]])
#
# dist = np.array([[ -3.94178951e-01, 1.96606002e-01, 2.01233145e-03, 5.16004605e-05,-5.30059696e-02]])
#
#
# images = glob.glob(dir + "/*.png")
#
#
#
# for fname in images
#     img = cv2.imread(fname)
#     h, w = img.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#     print newcameramtx
#     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#     # dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
#     x,y,w,h = roi
#     dst1 = dst[y:y+h,x:x+w]
#     # cv2.imwrite(dir+'/calibresult11.jpg', dst1)
#     print "size of dst:", dst1.shape
#     cv2.imshow('img', dst1)
#     cv2.waitKey(0)
#
