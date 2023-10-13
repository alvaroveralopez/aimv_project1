import cv2
import sys
import numpy as np

# GLOBAL VARIABLES
morph_size = 5
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# FUNCTIONS

# @function detectForeground - Detects the foreground in a given frame using a background model.

def detectForeground(frame, bgModel_RGB):
    """
    Detects the foreground in a given frame using a background model.

    Parameters:
    - frame: A numpy array representing the input frame.
    - bgModel: A background model to be applied on the frame.

    Returns:
    - Mask: A numpy array representing the foreground mask.
    """
    
    # Apply the background model to the frame to extract the foreground
    Mask = bgModel_RGB.apply(frame)

    # Perform noise removal
    Mask = cv2.erode(Mask, kernel)
    Mask = cv2.dilate(Mask, kernel)
    
    # Fill holes in the mask
    morph_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_size + 1, 2 * morph_size + 1),
                                        (morph_size, morph_size))
    Mask = cv2.dilate(Mask, element)
    Mask = cv2.erode(Mask, element)
    
    return Mask


def main():

    # READ THE VIDEO STREAM FROM WEBCAM OR VIDEO IF IT IS PROVIDED IN CONSOLE
    console_params = sys.argv
    if len(console_params) < 2:
        capture = cv2.VideoCapture(0) # try to open webcam, this will attempt to open the first one found in the system
    elif len(console_params) == 2:  
        capture = cv2.VideoCapture(console_params[1]) # try to open string, this will attempt to open it as a video file
    else:
        sys.exit(" You need to provide only a video file or nothing for using a webcam")

    fps = capture.get(cv2.CAP_PROP_FPS) # get the frames per second of the video/webcam
    print( "Frames per second : ", fps)

    cv2.namedWindow("edges", 1)

    # construct the class for background subtraction
    bgModel_RGB = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=16, detectShadows=False) 

    while(1):
        ret, frame = capture.read()
        edges = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.GaussianBlur(edges, (7, 7), 1.5)
        edges = cv2.Canny(edges, 0, 30, 3)
    
        cv2.imshow('frame',frame)# showing the video frames
        cv2.imshow('edges',edges)# showing the video edges

        Mask_RGB = detectForeground(frame, bgModel_RGB)
        Mask_edges = cv2.Canny(Mask_RGB, 0, 30, 3)

        # Display the foreground masks
        window_name1 = "Foreground mask: press esc to quit"
        cv2.imshow(window_name1, Mask_RGB)  # To get rid of shadows, use Mask == 255 here.

        window_name2 = "Foreground edges: press esc to quit"
        cv2.imshow(window_name2, Mask_edges)  

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__== '__main__':
    main()