import numpy as np
import matplotlib.pyplot as plt
import cv2
from Tracking.SupportScripts.Projection.projectionFunctions import *

def main():
    print("Projection")

    # Lettura immagine con OpenCV
    image_file = "src/Tracking/SupportScripts/Projection/imgs/test2.png"
    img = cv2.imread(image_file)



    # Parametri per test1
    x_real = np.array([-2.5, 0.5, 0.5, 4.6])
    y_real = np.array([13.41, 8.00, 13.00, 10.91])
    z_real = np.zeros_like(x_real)
    xt= 0
    yt= 0
    zt= 7.20  # Room coordinates
    thyaw = 0* np.pi / 180  # Z
    thpitch = (((360-32) * np.pi) / 180)  # X
    throll = 0 * np.pi / 180  # Y
    f = 0.003  # Focal distance (m)

    # parametri per test2
    if (image_file == "src/Projection/imgs/test2.png"):
        x_real = np.array([-3.5, 3.6, -2.8, 2])
        y_real = np.array([13.11, 10.61, 4.11, 5.81])
        z_real = np.zeros_like(x_real)
        xt= 0
        yt= 0
        zt= 6.92  # Room coordinates
        thyaw = 12 * np.pi / 180  # Z
        thpitch = ((-36) * np.pi) / 180  # X
        throll = 10.5 * np.pi / 180  # Y
        f = 0.00325  # Focal distance (m)

    # Fixed parameters
    s_w = 0.00498  # Sensor width (m)
    s_h = 0.00374  # Sensor height (m)
    U = img.shape[1]   # Image width (pixels)
    V = img.shape[0]  # Image height (pixels)


    img_with_points=inversion_draw_points(image=img, x_real=x_real, y_real=y_real, z_real=z_real, camera_x=xt, camera_y=yt, camera_z=zt, thyaw=thyaw, thpitch=thpitch, throll=throll, focal=f, resolution_x=U, resolution_y=V, sensor_x=s_w, sensor_y=s_h)


    cv2.imshow("Projected Points", img_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()