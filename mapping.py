import numpy as np
import matplotlib.pyplot as plt
import cv2

# Dati iniziali
x_real = np.array([-2.5, 0.5, 0.5, 4.6])
y_real = np.array([13.41, 8.00, 13.00, 10.91])
z_real = np.zeros_like(x_real)  # Assumendo z_real = 0 per i punti nel piano xy

xc, yc, zc = 0, 0, 7.20  # Coordinate della camera
thyaw = 0 * np.pi / 180  # Yaw (rotazione attorno a Z)
thpitch = -32 * np.pi / 180  # Pitch (rotazione attorno a Y)
throll = 0 * np.pi / 180  # Roll (rotazione attorno a X)
# Parametri immagine
f = 0.003  # Distanza focale (m)
s_w = 0.0049  # Larghezza sensore (m)
s_h = 0.00374  # Altezza sensore (m)
U = 1280  # Larghezza immagine (pixel)
V = 720  # Altezza immagine (pixel)


image_file = "./imgs/test20241212.png"
image = cv2.imread(image_file)
f_h=2*np.arctan(s_h/(2*f)) #fov verticale
f_w=2*np.arctan(s_w/(2*f)) #fov orizzontale

y_shift=zc*np.tan(thpitch+f_h/2)
dz = zc * np.tan(thpitch - f_h/2)
y_first=zc*np.tan(f_h/2)

diff = y_shift - y_first
print("diff ",diff)
print("dz ",dz)
print("f_v ", np.rad2deg(f_w))


for i in range(len(x_real)):
    z_real[i] = 7.20
    # y_real[i] = y_real[i] + diff
    z_real[i] = z_real[i] - diff

    
    d=np.sqrt(np.square(x_real[i])+np.square(y_real[i])+np.square(z_real[i]))
    H=2*d*np.tan(f_h/2) 
    W=2*d*np.tan(f_w/2)
    PPM_w=U/W
    PPM_h=V/H
    # x_real[i] = f*(x_real[i] / z_real[i])
    # y_real[i] = f*(y_real[i] / z_real[i])
    point = (int(U/2+(x_real[i]*PPM_w)), int((V - y_real[i]*PPM_h)))
    cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # Cerchi rossi

#W E H aumentano con l'aumentare della distanza


output_file = "esercitazione/imgs/output_image.png"
cv2.imwrite(output_file, image)
cv2.imshow("Projected Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()