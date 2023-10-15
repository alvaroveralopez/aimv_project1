import cv2 as cv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import math

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

# --------------------------------------------------------------------------------------------
# |                                      CONSTANTS                                           |
# --------------------------------------------------------------------------------------------
standard_name = "Standard Hough Lines Demo"
probabilistic_name = "Probabilistic Hough Lines Demo"
min_threshold = 10
max_trackbar = 170
alpha = 1000
csv = "database.csv"

h_bins = 30
s_bins = 32
histSize = [h_bins, s_bins]

h_ranges = [0, 180] # hue varies from 0 to 179,
s_ranges = [0, 256] # saturation from 0 to 255
ranges = h_ranges + s_ranges # concat lists

channels = [0, 2] # Use the 0-th and 1-st channels

# --------------------------------------------------------------------------------------------
# |                                      FUNCTIONS                                           |
# --------------------------------------------------------------------------------------------
def DrawHist_HS(Hist_HS, DisplayName):

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
            BGR = cv.cvtColor(np.uint8([[[H, binVal,S]]]), cv.COLOR_HLS2BGR)
            color = (round(BGR[0,0,0])*10, round(BGR[0,0,1])*10, round(BGR[0,0,2])*10) # I am multiplying by an arbitrary value to visualize colors better, because the weight of the black pixels is too high in the histogram
            start_point = (i*scale, j*scale)
            end_point = ((i+1)*scale, (j+1)*scale)
            hist2DImg = cv.rectangle(hist2DImg, start_point, end_point, color, thickness)

    y=np.flipud(hist2DImg) #turning upside down the image to have (0,0) in the lower left corner
    cv.imshow(DisplayName, y)

    return(0)

def remove_black_areas(image):
    # Convertir la imagen a escala de grises
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Encontrar los contornos en la imagen
    contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar el contorno más grande (rectángulo)
        largest_contour = max(contours, key=cv.contourArea)

        # Encontrar los vértices del rectángulo mínimo
        rect = cv.minAreaRect(largest_contour)

        # Calcular la matriz de transformación para rotar
        center, size, angle = rect
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # Rotar la imagen
        rotated_image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Obtener el ancho y alto del rectángulo rotado
        width, height = size

        # Recortar la región del rectángulo rotado
        x, y = int(center[0] - width / 2), int(center[1] - height / 2)
        x, y, w, h = int(x), int(y), int(width), int(height)

        cropped_image = rotated_image[y:y+h, x:x+w]

        return cropped_image

    # Si no se encontraron contornos, devolver una copia de la imagen original
    return image

def make_mask(imagen):
    img_name = os.path.basename(imagen)
    img_name = img_name.split(".")[0]

    img = cv.imread(imagen, cv.IMREAD_GRAYSCALE)
    imgRGB = cv.imread(imagen)

    blur = cv.GaussianBlur(img, (5, 5), 0)  # apply blur to grayscale image
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges = cv.Canny(blur, 100, 200)
    kernel = np.ones((17, 17), np.uint8)
    dilate = cv.dilate(edges, kernel, iterations=5)
    erode = cv.erode(dilate, kernel, iterations=5)

    final_mask = erode * img

    masked_b = cv.bitwise_and(imgRGB[:, :, 0], erode)
    masked_g = cv.bitwise_and(imgRGB[:, :, 1], erode)
    masked_r = cv.bitwise_and(imgRGB[:, :, 2], erode)
    final_img = cv.merge((masked_b, masked_g, masked_r))

    output_directory = "excImages"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cropped_img = remove_black_areas(final_img)
    cv.imwrite(f'excImages/{img_name}.png', cropped_img)

    # cv.imshow('Final image', final_img)
    return cropped_img

def make_hist(imagen):
    img = make_mask(imagen)

    if img is None:
        print("Could not read the image ", imagen)
        sys.exit()

    img = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    hist = cv.calcHist([img], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    # DrawHist_HS(hist, "Hue-Saturation Histogram")
    return hist

def debug_mode(img_path = None):
    print("\n\033[1;34m >>> You are currently in debugging mode\033[0m")

    # how many models there are in the DataBase
    if not os.path.exists(csv):
        print("\n       > There are no models in the database.")
    else:
        df = pd.read_csv(csv)
        num_models = len(df)

        if (num_models == 1):
            print(f"\n      There is {num_models} model in the database:\n")
        else:
            print(f"\n      There are {num_models} models in the database:\n")
        for i in range(df.shape[0]):
            model_i = df.at[i, 'name']
            print(f'            > {model_i}')

    # the min and max distances among them

    # the distances of the new model to the existing models in the DataBase

    if (img_path is not None and os.path.exists(csv)):
        df = pd.read_csv(csv)
        hist = make_hist(img_path)

        for i in range(df.shape[0]):
            # Accede al valor de 'hist_path' en la fila actual
            hist_path_i = df.at[i, 'hist_path']
            hist_i = np.load(hist_path_i)

            comp0 = cv.compareHist(hist, hist_i, method=0)
            comp1 = cv.compareHist(hist, hist_i, method=1)
            comp2 = cv.compareHist(hist, hist_i, method=2)
            comp3 = cv.compareHist(hist, hist_i, method=3)

            distance = cv.norm(hist, hist_i, cv.NORM_L2)

            name_i = df.at[i, 'name']
            print(f'\n--------------------------------------------------------'
                  f'\n\033[35mDistances to {name_i} are:\033[0m'
                  f'\nCorrelation:   {comp0}'
                  f'\nChiSquare:     {comp1}'
                  f'\nIntersection:  {comp2}'
                  f'\nBhattacharyya: {comp3}'
                  f'\nL2:            {distance}')

def run_program(img_path):
    if not os.path.exists(csv):
        # ------------------------------------
        #           CREATE DATABASE
        # ------------------------------------
        print("\nThere is no database for comparisons yet...")
        tetra_name = input("In order to create it, please, introduce the name of your tetrabrick: ")

        hist = make_hist(img_path)

        if not os.path.exists("Histograms"):
            os.mkdir("Histograms")

        hist_path = f'Histograms/{tetra_name}.npy'
        np.save(hist_path, hist)

        data = {"name": tetra_name,
                "path": img_path,
                "hist_path": hist_path}

        df = pd.DataFrame(data, index=[0])
        df[['name', 'path', 'hist_path']].to_csv(csv, index=False)
        print(f'\nThanks!, database created and saved in {csv}.')
        # ------------------------------------
    else:
        # Si el archivo existe, cargar los datos en un DataFrame
        df = pd.read_csv(csv)

        hist = make_hist(img_path)

        match = False
        for i in range(df.shape[0]):
            # Accede al valor de 'hist_path' en la fila actual
            hist_path_i = df.at[i, 'hist_path']
            hist_i = np.load(hist_path_i)

            comp0 = cv.compareHist(hist, hist_i, method=0)
            comp1 = cv.compareHist(hist, hist_i, method=1)
            comp2 = cv.compareHist(hist, hist_i, method=2)
            comp3 = cv.compareHist(hist, hist_i, method=3)

            if (comp1 < 100 and comp3 < 0.5):
                name = df.at[i, 'name']
                print(f'\nThat is a \033[1;34m{name}\033[0m tetrabrick')
                match = True

        if (not match):
            print("\nThere is no match for that tetrabrick...")
            tetra_name = input("Please, introduce the name of the new tetrabrick to register: ")
            hist_path = f'Histograms/{tetra_name}.npy'
            np.save(hist_path, hist)
            new_row = {'name': tetra_name,
                       'path': img_path,
                       'hist_path': hist_path}
            df.loc[len(df)] = new_row

            df = df.sort_values(by='name', ascending=True)
            # data.append(new_row, ignore_index=True)
            df[['name', 'path', 'hist_path']].to_csv(csv)
            print(f'{tetra_name} saved in the database {csv}')


# --------------------------------------------------------------------------------------------
# |                                      MAIN                                                |
# --------------------------------------------------------------------------------------------

def main():
    debug = False
    print("\n-----------------------------------------------"
          "\n       WELCOME TO TETRABRICK DETECTOR"
          "\n-----------------------------------------------")


    while True:
        if not debug:
            img_path = input("\nIntroduce an image to analyze (type 'exit' for ending or 'd' for entering in debug mode): ")
        else:
            img_path = input("\nIntroduce an image to analyze (type 'exit' for ending or 's' for going back to standard mode): ")

        if img_path.lower() == 'exit':
            print("\033[91mClosing the program, bye.\033[0m")
            break
        elif img_path.lower() == 'd':
            debug_mode()
            debug = True
        elif img_path.lower() == 's':
            debug = False
        else:
            if (debug):
                debug_mode(img_path)
                run_program(img_path)
            else:
                run_program(img_path)


if __name__== '__main__':
    main()