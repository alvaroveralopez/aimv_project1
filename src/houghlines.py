import cv2 as cv
import numpy as np
import sys
'''
import skimage
from functools import partial
from skimage import filters
'''
import math
import pandas as pn
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob
import os

global edges


# Global variables (be careful because variables in this list are used in the functions,
# so these functions can not be copied to another program as they are here...you would need to modify them)


window_name = "Edge Map"
standard_name = "Standard Hough Lines Demo"
min_threshold = 50
alpha = 1000


def Standard_Hough(edges):
    standard_hough = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # make a copy
    # Use Standard Hough Transform
    s_lines = cv.HoughLines(edges, 1, np.pi / 90, min_threshold)
    print(f"{s_lines}")

    # Show the result
    for line in s_lines:
        r = line[0][0]
        t = line[0][1]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        pt1 = (round(x0 - alpha * b), round(y0 + alpha * a))
        pt2 = (round(x0 + alpha * b), round(y0 - alpha * a))
        cv.line(standard_hough, pt1, pt2, (255, 0, 0), 2)
        print(f"Houghlines: Rho {r}, Theta: {t}, Point1: {pt1}, Point2: {pt2}")

    cv.imshow(standard_name, standard_hough)
    return s_lines


#########################
###  Main program #######
bgModel_RGB = cv.createBackgroundSubtractorMOG2()
imgRGB = cv.imread('C:/Users/alvar/PycharmProjects/aimv_project1/tetra/tcol7.bmp')

# print(f"\n\n\nimRGB: {imgRGB}\n\n\n")
edges = cv.cvtColor(imgRGB, cv.COLOR_BGR2GRAY)
edges1 = cv.GaussianBlur(edges, (31, 31), 4)
edges = cv.Canny(edges1, 40, 50, 3)
cv.imshow('frame', imgRGB)  # Mostrar los fotogramas del video
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow('edges', edges)  # Mostrar los bordes del video
cv.waitKey(0)
cv.destroyAllWindows()
a = edges
lines = Standard_Hough(edges)
print("holi")
cv.waitKey(0)

# Inicializar listas para almacenar las líneas paralelas y opuestas
parallel_lines = []
opposite_lines = []

# Tolerancia para considerar ángulos cercanos a 180 grados
angle_tolerance = 30

# Identificar líneas paralelas y opuestas
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        rho1, theta1 = lines[i][0]
        rho2, theta2 = lines[j][0]
        angle1_degrees = np.degrees(theta1)
        angle2_degrees = np.degrees(theta2)

        # Calcular la diferencia de ángulo
        angle_difference = abs(angle1_degrees - angle2_degrees)

        if abs(angle_difference - 180) < angle_tolerance:
            # Las líneas son opuestas
            opposite_lines.append((rho1, theta1, rho2, theta2))
        elif abs(angle_difference) < angle_tolerance:
            # Las líneas son paralelas
            parallel_lines.append((rho1, theta1, rho2, theta2))

# Seleccionar dos pares de líneas paralelas y opuestas
if len(parallel_lines) >= 2 and len(opposite_lines) >= 2:
    selected_parallel_pairs = parallel_lines[:2]
    selected_opposite_pairs = opposite_lines[:2]

    # Dibujar los rectángulos utilizando los pares de líneas seleccionados
    for pair in selected_parallel_pairs + selected_opposite_pairs:
        rho1, theta1, rho2, theta2 = pair
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        x1 = a1 * rho1
        y1 = b1 * rho1
        a2 = np.cos(theta2)
        b2 = np.sin(theta2)
        x2 = a2 * rho2
        y2 = b2 * rho2

        # Aquí puedes utilizar los puntos (x1, y1) y (x2, y2) para dibujar los rectángulos

# Mostrar la imagen con los rectángulos
cv.imshow('Rectángulos', imgRGB)
cv.waitKey(0)
cv.destroyAllWindows()