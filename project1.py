
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


standard_name = "Standard Hough Lines Demo"
probabilistic_name = "Probabilistic Hough Lines Demo"
min_threshold = 90
max_trackbar = 150
alpha = 1000



def Standard_Hough():

	standard_hough = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

	# Use Standard Hough Transform
	s_lines = cv.HoughLines(edges, 1, np.pi / 90, min_threshold)

	# s_lines = cv.HoughLines( edges, 3, np.pi/90, min_threshold); // uncomment this and comment the other to fix the threshold in a final solution

	# Show the result

	for line in s_lines:
		r = line[0][0]
		t = line[0][1]
		a = np.cos(t)
		b = np.sin(t)
		x0 = a*r
		y0 = b*r
		pt1 = (round(x0 - alpha * b), round(y0 + alpha * a))
		pt2 = (round(x0 + alpha * b), round(y0 - alpha * a))
		cv.line(standard_hough, pt1, pt2, (255, 0, 0), 2)


	cv.imshow(standard_name, standard_hough);

#
# @function Probabilistic_Hough
#
def Probabilistic_Hough():

	probabilistic_hough = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

	# Use Probabilistic Hough Transform
	p_lines = cv.HoughLinesP(edges, 1, np.pi/90, threshold=min_threshold, minLineLength=30, maxLineGap=30)

	# Show the result

	for points in p_lines:
		pt1 = (points[0][0], points[0][1])
		pt2 = (points[0][2], points[0][3])
		cv.line(probabilistic_hough, pt1, pt2, (255, 0, 0), 2)
		
	cv.imshow(probabilistic_name, probabilistic_hough)







img = cv.imread('tcol1.bmp', cv.IMREAD_GRAYSCALE)
blur = cv.GaussianBlur(img,(5,5),0)
assert img is not None, "file could not be read, check with os.path.exists()"
global edges
edges = cv.Canny(blur,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


kernel = np.ones((5,5),np.uint8)

dilation = cv.dilate(edges,kernel,iterations = 1)
plt.imshow(dilation,cmap = 'gray')
plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.show()


ret, thresh = cv.threshold(dilation, 127, 255, 0)
contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Encuentra el rectángulo mínimo delimitador
if len(contours) > 0:
    rect = cv.minAreaRect(contours[0])  # Suponiendo que estás interesado en el primer contorno
    box = cv.boxPoints(rect)
    box = np.intp(box)

    # Dibuja el rectángulo en la imagen original
    cv.drawContours(dilation, [box], -1, (0, 0, 255), 2)

# Muestra la imagen con el rectángulo
cv.imshow('Image with Rectangle', dilation)
cv.waitKey(0)
cv.destroyAllWindows()

Standard_Hough()
cv.waitKey(0)
Probabilistic_Hough()
cv.waitKey(0)
