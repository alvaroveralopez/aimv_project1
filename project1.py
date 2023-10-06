import cv2 as cv
import numpy as np
import sys
import skimage
from functools import partial  
from skimage import filters
import math
import pandas as pn
from matplotlib import pyplot as plt


standard_name = "Standard Hough Lines Demo"
probabilistic_name = "Probabilistic Hough Lines Demo"
min_threshold = 10
max_trackbar = 170
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


def recortar_areas_negras(imagen):
    # Convierte la imagen a escala de grises
    img_gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    # Define un umbral para detectar píxeles negros
    threshold_value = 10  # Ajusta este valor según tu imagen
    _, binary_mask = cv.threshold(img_gray, threshold_value, 255, cv.THRESH_BINARY)

    # Encuentra el contorno de las áreas negras
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Encuentra el rectángulo delimitador más grande de las áreas negras
    largest_area = 0
    largest_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    if largest_contour is not None:
        # Obtiene las coordenadas del rectángulo delimitador
        x, y, w, h = cv.boundingRect(largest_contour)

        # Recorta la imagen original para eliminar las áreas negras
        imagen_recortada = imagen[y:y+h, x:x+w]

        return imagen_recortada
    else:
        return None  # Si no se encontraron áreas negras, devuelve None



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
            BGR = cv.cvtColor(np.uint8([[[H,binVal,S]]]), cv.COLOR_HLS2BGR)
            color = (round(BGR[0,0,0])*10, round(BGR[0,0,1])*10, round(BGR[0,0,2])*10) # I am multiplying by an arbitrary value to visualize colors better, because the weight of the black pixels is too high in the histogram
            start_point = (i*scale, j*scale)
            end_point = ((i+1)*scale, (j+1)*scale)
            hist2DImg = cv.rectangle(hist2DImg, start_point, end_point, color, thickness)

    y=np.flipud(hist2DImg) #turning upside down the image to have (0,0) in the lower left corner
    cv.imshow(DisplayName,y)

    return(0)


img = cv.imread('tcol1.bmp', cv.IMREAD_GRAYSCALE)
imgRGB = cv.imread('tcol1.bmp')
blur = cv.GaussianBlur(img,(5,5),0)
assert img is not None, "file could not be read, check with os.path.exists()"
global edges
edges = cv.Canny(blur,100,200)


kernel3 = np.ones((3,3),np.uint8)
kernel5 = np.ones((5,5),np.uint8)
kernel7 = np.ones((7,7),np.uint8)
kernel9 = np.ones((9,9),np.uint8)
kernel11 = np.ones((11,11),np.uint8)
kernel15 = np.ones((15,15),np.uint8)
kernel17 = np.ones((17,17),np.uint8)

#https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
#opening = cv.morphologyEx(edges, cv.MORPH_GRADIENT, kernel17)
#opening = cv.dilate(opening, kernel11, iterations = 1)
#opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel17)
dilate = cv.dilate(edges,kernel17,iterations = 5)
erode = cv.erode(dilate,kernel17,iterations = 5)

final_mask=erode*img
""" output_img = np.zeros_like(final_mask)
output_img[final_mask == 255] = [0, 0, 0]  # [B, G, R] values for black """



# Crea una figura de Matplotlib con una cuadrícula de 2x2 y muestra cada imagen en una ubicación específica
plt.subplot(221)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.axis('off')

plt.subplot(223)
plt.imshow(erode, cmap='gray')
plt.title('Opening')
plt.axis('off')

# Deja el cuarto subplot en blanco (si deseas agregar una cuarta imagen, puedes hacerlo aquí)
plt.subplot(223)
plt.imshow(final_mask, cmap='gray')
plt.title('Wish image')
plt.axis('off')

# Ajusta automáticamente el espacio entre subplots
plt.tight_layout()
# Muestra la figura con todas las imágenes
plt.show()

masked_b = cv.bitwise_and(imgRGB[:, :, 0], erode)
masked_g = cv.bitwise_and(imgRGB[:, :, 1], erode)
masked_r = cv.bitwise_and(imgRGB[:, :, 2], erode)
# Combina los canales para formar la imagen resultante
final_final_img = cv.merge((masked_b, masked_g, masked_r))

cropped_img=recortar_areas_negras(final_final_img)
nombre_archivo = 'imagen_guardada.jpg'  # Cambia el nombre y la extensión del archivo según tu preferencia
cv.imwrite(nombre_archivo, cropped_img)

cv.imshow('Final image', final_final_img)
cv.waitKey(0)

cv.destroyAllWindows()

cropped_img=recortar_areas_negras(final_final_img)
nombre_archivo = 'imagen_guardada.jpg'  # Cambia el nombre y la extensión del archivo según tu preferencia
cv.imwrite(nombre_archivo, cropped_img)

cv.waitKey(0)

cv.destroyAllWindows()


#Making the histogram
img_descriptor_list = cv.calcHist(cropped_img, channels, None, histSize, ranges, accumulate=False)
cv.normalize(img_descriptor_list, img_descriptor_list, alpha=0, beta=1, norm_type=cv.NORM_MINMAX )
DrawHist_HS( img_descriptor_list, 'Histogram')
cv.waitKey(0)
cv.destroyAllWindows()


