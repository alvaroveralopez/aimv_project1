import cv2 as cv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
import skimage
from functools import partial
from skimage import filters
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import glob

global edges

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.cbook


# --------------------------------------------------------------------------------------------
# |                                      CONSTANTS                                           |
# --------------------------------------------------------------------------------------------
standard_name = "Standard Hough Lines Demo"
probabilistic_name = "Probabilistic Hough Lines Demo"
min_threshold = 10
max_trackbar = 170
alpha = 1000
DATABASE_FILE = "database.csv"

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
def cut_black_areas(imagen):
    img_gray = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    threshold_value = 10
    _, binary_mask = cv.threshold(img_gray, threshold_value, 255, cv.THRESH_BINARY)

    # Encontrar el contorno más grande
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    largest_contour = None

    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    if largest_contour is not None:
        # Calcular el ángulo de rotación del rectángulo delimitador
        rect = cv.minAreaRect(largest_contour)
        angle = rect[2]

        if angle < -45:
            angle += 90

        # Rotar la imagen para enderezar el tetrabrick
        rows, cols, _ = imagen.shape
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        rotated_image = cv.warpAffine(imagen, M, (cols, rows))

        # Recortar la imagen rotada original para eliminar las áreas negras
        rotated_gray = cv.cvtColor(rotated_image, cv.COLOR_BGR2GRAY)
        _, rotated_binary_mask = cv.threshold(rotated_gray, threshold_value, 255, cv.THRESH_BINARY)
        rotated_contours, _ = cv.findContours(rotated_binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        largest_contour = None

        for contour in rotated_contours:
            area = cv.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        if largest_contour is not None:
            x, y, w, h = cv.boundingRect(largest_contour)
            imagen_recortada = rotated_image[y:y+h, x:x+w]

            return imagen_recortada

    return None  # Si no se encontraron áreas negras o no se pudo corregir el giro, devuelve None



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

def make_mask(imagen):
    img_name = os.path.basename(imagen)
    img_name = img_name.split(".")[0]

    img = cv.imread(imagen, cv.IMREAD_GRAYSCALE)
    imgRGB = cv.imread(imagen)

    blur = cv.GaussianBlur(img, (5, 5), 0) # apply blur to grayscale image
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges = cv.Canny(blur, 100, 200)
    kernel = np.ones((17, 17), np.uint8)
    dilate = cv.dilate(edges, kernel, iterations=5)
    erode = cv.erode(dilate, kernel, iterations=5)

    final_mask = erode * img

    # Figura ----------------------------------------------------------------------
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

    plt.subplot(224)
    plt.imshow(final_mask, cmap='gray')
    plt.title('Wish image')
    plt.axis('off')

    plt.suptitle(f'{img_name}', fontsize=16)

    plt.tight_layout()
    # plt.show(block=False)

    # Esperar a que se presione una tecla
    # plt.waitforbuttonpress()
    # ------------------------------------------------------------------------

    masked_b = cv.bitwise_and(imgRGB[:, :, 0], erode)
    masked_g = cv.bitwise_and(imgRGB[:, :, 1], erode)
    masked_r = cv.bitwise_and(imgRGB[:, :, 2], erode)
    final_img = cv.merge((masked_b, masked_g, masked_r))

    output_directory = "excImages"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cropped_img = cut_black_areas(final_img)
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
    
def debug_mode(img_path=None):
    print("\n\033[1;34m >>> You are currently in debugging mode\033[0m")

    if not os.path.exists(DATABASE_FILE):
        print("\n       > There are no models in the database.")
    else:
        df = pd.read_csv(DATABASE_FILE)
        num_models = len(df)

        if num_models == 1:
            print(f"\n      There is {num_models} model in the database:\n")
        else:
            print(f"\n      There are {num_models} models in the database:\n")

        for model_name in df['name']:
            print(f'            > {model_name}')

    if img_path is not None and os.path.exists(DATABASE_FILE):
        df = pd.read_csv(DATABASE_FILE)
        hist = make_hist(img_path)

        for _, row in df.iterrows():
            hist_i = np.load(row['hist_path'])
            comp0 = cv.compareHist(hist, hist_i, method=0)
            comp1 = cv.compareHist(hist, hist_i, method=1)
            comp2 = cv.compareHist(hist, hist_i, method=2)
            comp3 = cv.compareHist(hist, hist_i, method=3)
            distance = cv.norm(hist, hist_i, cv.NORM_L2)
            name_i = row['name']

            print(f'\n--------------------------------------------------------'
                  f'\n\033[35mDistances to {name_i} are:\033[0m'
                  f'\nCorrelation:   {comp0}'
                  f'\nChiSquare:     {comp1}'
                  f'\nIntersection:  {comp2}'
                  f'\nBhattacharyya: {comp3}'
                  f'\nL2:            {distance}')

def run_program(img_path):
    if os.path.exists(img_path):
        if not os.path.exists(DATABASE_FILE):
            print("\nThere is no database for comparisons yet...")
            tetra_name = input("In order to create it, please, introduce the name of your tetrabrick: ")

            hist = make_hist(img_path)

            if not os.path.exists("Histograms"):
                os.mkdir("Histograms")

            hist_path = f'Histograms\{tetra_name}.npy'
            np.save(hist_path, hist)

            data = {"name": tetra_name, "path": img_path, "hist_path": hist_path}
            df = pd.DataFrame(data, index=[0])
            df.to_csv(DATABASE_FILE, index=False)
            print(f'\nThanks! Database created and saved in {DATABASE_FILE}.')
        else:
            df = pd.read_csv(DATABASE_FILE)
            hist = make_hist(img_path)
            match = False

            for _, row in df.iterrows():
                hist_i = np.load(row['hist_path'])
                comp1 = cv.compareHist(hist, hist_i, method=1)
                comp3 = cv.compareHist(hist, hist_i, method=3)

                if comp1 < 100 and comp3 < 0.5:
                    name = row['name']
                    print(f'\nThat is a \033[1;34m{name}\033[0m tetrabrick')
                    match = True

            if not match:
                print("\nThere is no match for that tetrabrick...")
                tetra_name = input("Please, introduce the name of the new tetrabrick to register: ")
                hist_path = f'Histograms\{tetra_name}.npy'
                np.save(hist_path, hist)
                new_row = {'name': tetra_name, 'path': img_path, 'hist_path': hist_path}
                df = df.append(new_row, ignore_index=True)
                df = df.sort_values(by='name', ascending=True)
                df.to_csv(DATABASE_FILE, index=False)
                print(f'{tetra_name} saved in the database {DATABASE_FILE}')
    else:
        print(f'\nFile \033[1;31m{img_path}\033[0m not found.')

def run_program2(directory):
    try:
        for file in os.listdir(directory):
            img_path = os.path.join(directory, file)
            img = cv.imread(img_path)

            if img is not None and img.shape[0] > 0 and img.shape[1] > 0: # Si se cumple el tamaño de la imagen, el último bucle daría un warning
                cv.imshow("Current Image", img)
                cv.waitKey(0)  # Wait for a key press
                cv.destroyAllWindows()

                if not os.path.exists(DATABASE_FILE):
                    print("\nThere is no database for comparisons yet...")
                    
                    tetra_name = input("In order to create it, please, introduce the name of your tetrabrick: ")

                    hist = make_hist(img_path)

                    if not os.path.exists("Histograms"):
                        os.mkdir("Histograms")

                    hist_path = f'Histograms\{tetra_name}.npy'
                    np.save(hist_path, hist)

                    data = {"name": tetra_name, "path": img_path, "hist_path": hist_path}
                    df = pd.DataFrame(data, index=[0])
                    df.to_csv(DATABASE_FILE, index=False)
                    print(f'\nThanks! Database created and saved in {DATABASE_FILE}.')
                else:
                    df = pd.read_csv(DATABASE_FILE)
                    hist = make_hist(img_path)
                    match = False

                    for _, row in df.iterrows():
                        hist_i = np.load(row['hist_path'])
                        comp1 = cv.compareHist(hist, hist_i, method=1)
                        comp3 = cv.compareHist(hist, hist_i, method=3)

                        if comp1 <= 100 and comp3 < 0.5:
                            name = row['name']
                            print(f'\nThat is a \033[1;34m{name}\033[0m tetrabrick')
                            match = True

                    if not match:
                        print("\nThere is no match for that tetrabrick...")
                        tetra_name = input("Please, introduce the name of the new tetrabrick to register: ")
                        hist_path = f'Histograms\{tetra_name}.npy'
                        np.save(hist_path, hist)
                        new_row = {'name': tetra_name, 'path': img_path, 'hist_path': hist_path}
                        df = df._append(new_row, ignore_index=True)
                        df = df.sort_values(by='name', ascending=True)
                        df.to_csv(DATABASE_FILE, index=False)
                        print(f'{tetra_name} saved in the database {DATABASE_FILE}')
    except Exception as e:
        print("Error:", e)

def main():
    try:
        debug = False

        # Create a tkinter root window (hidden), para que salga la ventana de elegir carpeta encima de todo
        root = tk.Tk()
        root.attributes("-topmost", True)
        root.withdraw()

        print("\n-----------------------------------------------"
            "\n       WELCOME TO TETRABRICK DETECTOR"
            "\n-----------------------------------------------")

        code_folder = os.path.dirname(os.path.abspath(__file__))
        print(code_folder)

        while True:
            if not debug:
                img_path = input(f"\nIntroduce an image to analyze (type 'exit' for ending, 'd' for entering in debug mode or 'all' for analyzing all images): ")
            else:
                img_path = input(f"\nIntroduce an image to analyze (type 'exit' for ending, 's' for going back to standard mode or 'all' for analyzing all images): ")

            img_path = img_path.replace("/", "\\")

            if img_path.lower() == 'exit':
                print("\033[91mClosing the program, bye.\033[0m")
                break
            elif img_path.lower() == 'd':
                debug_mode()
                debug = True
            elif img_path.lower() == 's':
                debug = False
            elif img_path.lower() == 'all':
                root = tk.Tk()
                root.withdraw()
                # Prompt the user to select a directory
                directory = filedialog.askdirectory()
                run_program2(directory)
                debug = False
            else:
                img_path = os.path.join(code_folder, img_path)
                if debug:
                    debug_mode(img_path)
                    run_program(img_path)
                else:
                    run_program(img_path)
        root.destroy()
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
