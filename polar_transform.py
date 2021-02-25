import cv2
import numpy as np 
import sys
import math 

def polar(I, center, r, theta= (0,360), rstep = 1.0, thetastep = 360.0/(180*6.8)):
	#得到距离最小最大的范围
	minr, maxr = r
	#得到角度最小最大范围
	mintheta, maxtheta = theta
	#
	H = int((maxr - minr) / rstep) +1
	W = int((maxtheta - mintheta) / thetastep) +1
	O = 125 * np.ones((H, W), I.dtype)

	#tile(a,(2,3))在垂直方向和水平方向上复制2，3次
	

	#numpy.linspace(start, stop[, num=50[, endpoint=True[, retstep=False[, dtype=None]]]]])
	#返回在指定范围内的均匀间隔的数字（组成的数组），也即返回一个等差数列
	#start - 起始点，stop - 结束点，num - 元素个数，默认为50，
	#endpoint - 是否包含stop数值，默认为True，包含stop值；若为False，则不包含stop值
	#retstep - 返回值形式，默认为False，返回等差数列组，若为True，则返回结果(array([`samples`, `step`])),
	#dtype - 返回结果的数据类型，默认无，若无，则参考输入数据类型。

	#transpose用法详见https://blog.csdn.net/xiongchengluo1129/article/details/79017142

	#极坐标变换
	r = np.linspace(minr,maxr,H)			
	r = np.tile(r,(W,1))
	r = np.transpose(r)
	theta = np.linspace(mintheta,maxtheta,W)
	theta = np.tile(theta,(H,1))
	x,y=cv2.polarToCart(r,theta,angleInDegrees=True)


	#最近邻插值
	for i in range(H):
		for j in range(W):
			px = int(round(x[i][j])+cx)
			py = int(round(y[i][j])+cy)
			if((px >= 0 and px <= w-1) and (py >= 0 and py <= h-1)):
				O[i][j] = I[py][px]
			else:
				O[i][j] = 125#灰色
	return O
    
#主函数
if __name__ == "__main__":
	imagePath = "/Users/wongtyu/Downloads/cvusa/bingmap/18/0000054.jpg"  #"G:\\blog\\OpenCV算法精解-测试图片\\第3章\\image2.jpg"
	image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	
    #图像的宽高
	h,w = image.shape[:2]
	print (w,h)
    # #极左标变换的中心
	#极左标变换的中心
	cx,cy = 112,112
	print (cx,cy)
	cv2.circle(image,(int(cx),int(cy)),37,(255.0,0,0),3)
	cv2.circle(image,(int(cx),int(cy)),74,(255.0,0,0),3)
	# cv2.rectangle(image,(50,50),(162,162),(55,255,155),5)
    #距离的最小最大半径 #200 550 270,340
	O = polar(image,(cx,cy),(0,115))
    #旋转
	O = cv2.flip(O,0)
    #显示原图和输出图像
	cv2.imshow("image",image)
	cv2.imshow("O",O)
	cv2.imwrite("O.jpg",O)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
