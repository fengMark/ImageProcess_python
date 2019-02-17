# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
image=cv2.imread('image.bmp')
weigth_im=image.shape[1];
heigth_im=image.shape[0];
number=image.shape[2]
image=cv2.resize(image,(weigth_im*2,heigth_im*2),cv2.INTER_LINEAR);
weigth_im=image.shape[1];
heigth_im=image.shape[0];
number=image.shape[2]
print("原图像大小：\n""weight: %d \nheight: %d \nnumber: %d" %(weigth_im,heigth_im,number))
def main(argv):
#    if len(sys.argv)>1:
#        image = cv2.imread(sys.argv[1])
#    else:
#        print("Usage:python oper.py imageFile")
    cv2.imshow("image",image)
    cv2.waitKey(0)
    newimage=Gamma(image)
    newimage=GaussianBlurOper(newimage)
#    DilateOper(image)
    newimage=LaplaceOper(newimage)
    newimage=ErodeOper(newimage)
    

    
def GaussianBlurOper(image):
    blurImage=cv2.GaussianBlur(image,(3,3),0)
    blurImage=np.round(blurImage)
    blurImage=blurImage.astype(np.uint8) 
    cv2.imshow("GaussBlur",blurImage)
    cv2.imwrite("GaussinBlurOper.bmp",blurImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows();
    return blurImage
    
def DilateOper(image):#膨胀图像
    r=1
    MAX_R=20
    cv2.namedWindow("dilate",1)
    def nothing(*arg):
        pass
    cv2.createTrackbar("r","dilate",r,MAX_R,nothing);
    while True:
        r=cv2.getTrackbarPos('r','dilate')
        s=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
        opimage=cv2.dilate(image,s);
        cv2.imshow("dilate",opimage);
        cv2.imwrite("DilateOper.bmp",opimage)
        ch=cv2.waitKey(100);
        if ch==27:
            break
    cv2.destroyAllWindows()
    return opimage
def ErodeOper(image):
    r=1
    MAX_R=20
    cv2.namedWindow("erode",1)
    def nothing(*arg):
        pass
    cv2.createTrackbar("r","erode",r,MAX_R,nothing);
    while True:
        r=cv2.getTrackbarPos('r','erode')
        s=cv2.getStructuringElement(cv2.MORPH_RECT,(2*r+1,2*r+1))
        opimage=cv2.erode(image,s)
        cv2.imshow("erode",opimage)
        cv2.imwrite("ErodeOper.bmp",opimage)
        ch=cv2.waitKey(100);
        if ch==27:
            break
    cv2.destroyAllWindows()
    return opimage
def LaplaceOper(image):#锐化图像
    weight=image.shape[0]
    height=image.shape[1]
    number=image.shape[2]
    grayimg=np.zeros((image.shape[0],image.shape[1],1),np.uint8)
    print("原图像大小：\n""weight: %d \nheight: %d \nnumber: %d" %(weight,height,number)) 
    for i in range(weight):
        for j in range(height):
            grayimg[i,j] = 0.299 * image[i,j,0] + 0.587 * image[i,j,1] + 0.114 * image[i,j,2]
    t1 = list([[0,1,0],
               [1,-4,1],
               [0,1,0]]) # 定义拉普拉斯滤波器
    shp=grayimg*1 # 设置一个新的图片变量，防止修改原图片
    shp=np.pad(grayimg,((1, 1), (1, 1),(0,0)),"constant",constant_values=0) # 为原图片加一层边缘
    for i in range(1,weight-1):
        for j in range(1,height-1):
            shp[i,j]=abs(np.sum(shp[i:i+3,j:j+3]*t1)) # 对灰度图进行锐化
    cv2.imshow('srcImage', image)
    cv2.imshow('grayImage', grayimg)
    cv2.imshow('grayImage', shp)
    cv2.imwrite("LaplaceOper.bmp",shp)
    cv2.imshow("Laplacian",grayimg+shp[1:shp.shape[0]-1,1:shp.shape[1]-1])
    cv2.imshow("Laplacian",image+shp[1:shp.shape[0]-1,1:shp.shape[1]-1])
    cv2.imwrite("LaplaceOper.bmp",grayimg+shp[1:shp.shape[0]-1,1:shp.shape[1]-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return grayimg+shp[1:shp.shape[0]-1,1:shp.shape[1]-1]
def Gamma(image):#对比度增强
    f1=image/255.0
    gamma=1
    MAX_GAMMA=1000
    cv2.namedWindow("Gamma",1)
    def nothing(*arg):
        pass
    cv2.createTrackbar("r","Gamma",gamma,MAX_GAMMA,nothing);
    while True:
        gamma=cv2.getTrackbarPos('r','Gamma')
        res=np.power(f1,gamma*0.02)
        cv2.imshow("Gamma",res)
        cv2.imwrite("Gamma.bmp",res)
        ch=cv2.waitKey(100);
        if ch==27:
            break
    cv2.destroyAllWindows()
    return res
if __name__=="__main__":
    main(sys.argv)
    
