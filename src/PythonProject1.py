import cv2 
import sys
import numpy as np
from functools import partial # This is needed to pass the original image to the Callback function (createTrackbar)

# Global variables (be careful because variables in this list are used in the functions, 
# so these functions can not be copied to another program as they are here...you would need to modify them)


window_name = "Edge Map"
standard_name = "Standard Hough Lines Demo"
probabilistic_name = "Probabilistic Hough Lines Demo"
min_threshold = 30
max_trackbar = 150
s_trackbar = 0
p_trackbar = 0

edgeThresh = 1
lowThreshold = 30
max_lowThreshold = 100
multiplier = 3
kernel_size = 3
alpha = 1000




h_bins = 30
s_bins = 32
histSize = [h_bins, s_bins]
# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0, 2]

#
# * @function CannyThreshold
# * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
#

def CannyThreshold(lowThreshold, a):
    global edges
    src_gray = cv2.cvtColor(a , cv2.COLOR_BGR2GRAY)
    # Reduce noise with a kernel 5x5
    blurred = cv2.blur(src_gray, (5, 5))
    # Canny detector
    edges = cv2.Canny(blurred, lowThreshold, lowThreshold*multiplier, kernel_size)
    Mask = edges/255
    # Using Canny's output as a mask, we display our result
    dst = a * (Mask[:,:,None].astype(a.dtype))
    cv2.imshow(window_name, dst)

#
# @function Standard_Hough
#
def Standard_Hough(s_trackbar):

	standard_hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

	# Use Standard Hough Transform
	s_lines = cv2.HoughLines(edges, 1, np.pi / 90, min_threshold+s_trackbar)

	# s_lines = cv2.HoughLines( edges, 3, np.pi/90, min_threshold); // uncomment this and comment the other to fix the threshold in a final solution

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
		cv2.line(standard_hough, pt1, pt2, (255, 0, 0), 2)


	cv2.imshow(standard_name, standard_hough);

#
# @function Probabilistic_Hough
#
def Probabilistic_Hough(p_trackbar):

	probabilistic_hough = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

	# Use Probabilistic Hough Transform
	p_lines = cv2.HoughLinesP(edges, 1, np.pi/90, threshold=min_threshold+ p_trackbar, minLineLength=30, maxLineGap=30)

	# Show the result

	for points in p_lines:
		pt1 = (points[0][0], points[0][1])
		pt2 = (points[0][2], points[0][3])
		cv2.line(probabilistic_hough, pt1, pt2, (255, 0, 0), 2)
		
	cv2.imshow(probabilistic_name, probabilistic_hough)

def DrawHist_HS( Hist_HS, DisplayName):

    bins1 = Hist_HS.shape[0]
    bins2 = Hist_HS.shape[1]
    scale = 10
    hist2DImg = np.zeros((bins1*scale, bins2*scale,3), dtype = np.uint8) # empty image of size bis1xbins2 and scaled to see the 2D histogram better
    thickness = -1
    for i in range(bins1):
        for j in range(bins2):
            binVal = np.uint8(Hist_HS[i, j]*255)
            # converting the histogram value to Intensity and using the corresponding H-S we can recover the RGB and visualize the histogram in color
            H = np.uint8(i/bins1*180 + h_ranges[0])
            S = np.uint8(j/bins2*255 + s_ranges[0])
            BGR = cv2.cvtColor(np.uint8([[[H,binVal,S]]]), cv2.COLOR_HLS2BGR)
            color = (round(BGR[0,0,0])*10, round(BGR[0,0,1])*10, round(BGR[0,0,2])*10) # I am multiplying by an arbitrary value to visualize colors better, because the weight of the black pixels is too high in the histogram
            start_point = (i*scale, j*scale)
            end_point = ((i+1)*scale, (j+1)*scale)
            hist2DImg = cv2.rectangle(hist2DImg, start_point, end_point, color, thickness)

    y=np.flipud(hist2DImg) #turning upside down the image to have (0,0) in the lower left corner
    cv2.imshow(DisplayName,y)

    return(0)

#########################
###  Main program #######
def main():

    #img_files = sys.argv
    img_files = ['program', 'tcol1.bmp', 'tcol2.bmp', 'tcol3.bmp']


    if len(img_files) <4:
        sys.exit(" You need to provide at least 3 image files")

    img_descriptor_list = {} # initializing a dictionary to collect the descriptor of images
    for index, filename in enumerate(img_files[1:4]):
        img = cv2.imread(filename) 
        if img is None:
            print("Could not read the image ", filename)
            sys.exit()

        cv2.imshow("Display window", img)    

        # Change of Color Space to calculate histograms in H-S
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Extracting the 2D histogram as an image descriptor

        img_descriptor_list[index] = cv2.calcHist([img], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(img_descriptor_list[index], img_descriptor_list[index], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX )
        DrawHist_HS(img_descriptor_list[index], "Hue-Saturation Histogram")
        cv2.waitKey(0)

    ###########
    # Visualizing Canny and Hough thresholds for detecting the external edges of tetra-bricks (only for the first image)

    img = cv2.imread(img_files[1]) 

    # Create a window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create a Trackbar for user to enter Canny threshold (comment it to use the default values defined in the header)

    cv2.createTrackbar("Min Threshold", window_name, lowThreshold, max_lowThreshold, partial(CannyThreshold, a=img))
    
    # Show the image
    CannyThreshold(lowThreshold, img)

    # Wait until user exit program by pressing a key
    cv2.waitKey(0)


    # Create Trackbar for Hough Threshold

    cv2.namedWindow(standard_name, cv2.WINDOW_NORMAL);

    cv2.createTrackbar("Hough threshold", standard_name, s_trackbar, max_trackbar, Standard_Hough)

    start = cv2.getTickCount()
    # Show the image
    Standard_Hough(s_trackbar)
    duration = (cv2.getTickCount()-start)/cv2.getTickFrequency()
    print("duration of Standard Hough process (msec) = ", duration)

    # Create Trackbar for Probabilistic Hough Threshold

    cv2.namedWindow(probabilistic_name, cv2.WINDOW_NORMAL);

    cv2.createTrackbar("Prob. Hough threshold", probabilistic_name, p_trackbar, max_trackbar, Probabilistic_Hough)

    start = cv2.getTickCount()
    # Show the image
    Probabilistic_Hough(p_trackbar)
    duration = (cv2.getTickCount()-start)/cv2.getTickFrequency()
    print("duration of Probabilistic Hough process (msec) = ", duration)

    # Wait until user exit program by pressing a key
    cv2.waitKey(0)

    #############

    print("Comparing the histograms of the 3 image files :")

    # Apply the histogram comparison methods
    # OpenCV does everything! See docs.opencv.org to understand the methods. You can program new histogram measurements
    # I am not using any segmentation here

    print("Matching Methods: 0=Correlation, 1=ChiSquare, 2=Intersection, 3=Bhattacharyya")
    for method in range(4):
        base_base = cv2.compareHist(img_descriptor_list[0], img_descriptor_list[0], method)
        base_test1 = cv2.compareHist(img_descriptor_list[0], img_descriptor_list[1], method)
        base_test2 = cv2.compareHist(img_descriptor_list[0], img_descriptor_list[2], method)
        print(f" Matching Method {method}: Perfect: {base_base}, Base-Test(1): {base_test1}, Base-Test(2): {base_test2} \n")

    print("Done \n")


if __name__== '__main__':
    main()