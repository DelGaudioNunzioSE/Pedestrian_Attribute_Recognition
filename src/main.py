import numpy as np
import matplotlib.pyplot as plt
import cv2
from Projection.file import *

def main():
    print("Projection")

    # Lettura immagine con OpenCV
    image_file = "src/Projection/test20241212.png"
    img = cv2.imread(image_file)



    # Dati iniziali
    x_real = np.array([-2.5, 0.5, 0.5, 4.6])
    y_real = np.array([13.41, 8.00, 13.00, 10.91])
    z_real = np.zeros_like(x_real)

    xt= 0
    yt= 0
    zt= 7.20  # Coordinate della camera
    thyaw = 0 * np.pi / 180  # Yaw (rotazione attorno a Z)
    thpitch = ((-32 * np.pi) / 180)  # Pitch (rotazione attorno a Y) (radianti)
    throll = 0 * np.pi / 180  # Roll (rotazione attorno a X)
    # Parametri immagine
    f = 0.003  # Distanza focale (m)
    s_w = 0.00498  # Larghezza sensore (m)
    s_h = 0.00374  # Altezza sensore (m)
    U = img.shape[1]   # Larghezza immagine (pixel)
    V = img.shape[0]  # Altezza immagine (pixel)


    img_with_points=draw_points(image=img, x_real=x_real, y_real=y_real, z_real=z_real, camera_x=xt, camera_y=yt, camera_z=zt, thyaw=thyaw, thpitch=thpitch, throll=throll, focal=f, resolution_x=U, resolution_y=V, sensor_x=s_w, sensor_y=s_h)


    # Salva e visualizza il risultato
    output_file = "./Projection/output_image.png"
    cv2.imwrite(output_file, img_with_points)

    cv2.imshow("Projected Points", img_with_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()