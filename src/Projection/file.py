import numpy as np
import matplotlib.pyplot as plt
import cv2


# Matrice di rotazione attorno all'asse Y

def R_z(theta_z):
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0,0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1 ,0],
        [0, 0, 0, 1]
    ])
    return Rz


def R_y(theta_y):
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y),0],
        [0, 1, 0,0],
        [-np.sin(theta_y), 0, np.cos(theta_y),0],
        [0, 0, 0, 1]
    ])
    return Ry

def R_x(theta_x):
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])
    return Rx



def C(x,y,z):
    C = np.array([
        [1,0,0,-x],
        [0,1,0,-y],
        [0,0,1,-z],
        [0,0,0,1]
    ])
    return C

def UVW(x,y,z):
    UVW = np.array([
        [x],
        [y],
        [z],
        [1]
    ])
    return UVW


def real_to_camera(point_x, point_y, point_z, camera_x, camera_y, camera_z,  theta_x=0, theta_y=0, theta_z=0):
    translation_matrix = C(camera_x, camera_y, camera_z)
    
    # Poi, applica la rotazione (intorno agli assi Z, Y, e X)
    rotation_matrix = R_z(theta_z) @ R_y(theta_y) @ R_x(theta_x)  # Assumiamo l'ordine di rotazione Z-Y-X
    
    # Ora combiniamo la trasformazione (rotazione + traslazione)
    transformation_matrix = rotation_matrix @ translation_matrix
    
    # Applichiamo la trasformazione al punto in coordinate UVW
    UVW_point = UVW(point_x, point_y, point_z)
    XYZ = transformation_matrix @ UVW_point

    print("rotation:")
    print(rotation_matrix)
    print("transformation_matrix:")
    print(transformation_matrix)
    print("XYZ:")
    print(XYZ)
    return XYZ




def focal_length(focal,s_resolution,p_sensor):
    s= p_sensor/s_resolution
    return focal/s


def camera_to_plane(XYZ,focal, resolution_x, resolution_y, sensor_x, sensor_y):

    fx = focal_length(focal, resolution_x, sensor_x)  # Calcolo della focale in x
    fy = focal_length(focal, resolution_y, sensor_y)  # Calcolo della focale in y

    ox = resolution_x / 2  # Centro dell'immagine in x
    oy = resolution_y / 2  # Centro dell'immagine in y

    # Matrice di proiezione
    perspective_projection = np.array([
        [fx, 0, ox, 0],
        [0, fy, oy, 0],
        [0, 0, 1, 0]
    ])

    # Moltiplicazione della matrice di proiezione per le coordinate 3D
    uvw = perspective_projection @ XYZ  # Risultato in coordinate omogenee (3x1)

    # Normalizzazione per ottenere le coordinate in 2D nel piano immagine
    x = uvw[0] / uvw[2]  # Coordinata x normalizzata
    y = uvw[1] / uvw[2]  # Coordinata y normalizzata

    xy = np.array([x, y])  # Coordinata omogenea normalizzata  


    print("fx:")
    print(fx)
    print("fy:")
    print(fy)
    print("perspective_projection:")
    print(perspective_projection)
    print("uvw:")
    print(uvw)
    print("xy:")
    print(xy)

    return xy





def plane_to_pixel(xy):
    u = xy[0]  # Coordinata u (pixel x)
    v = xy[1]  # Coordinata v (pixel y)
    return int(u[0]), int(v[0])






# Lettura immagine con OpenCV
image_file = "./test20241212.png"
image = cv2.imread(image_file)



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
U = image.shape[1]   # Larghezza immagine (pixel)
V = image.shape[0]  # Altezza immagine (pixel)

print("larghezza ed altezza immagine:")
print(U,V)


# Disegna i punti sull'immagine

for i in range(len(x_real)):
    print("point:")
    print(x_real[i],z_real[i], y_real[i])

    XYZ=real_to_camera(point_x=x_real[i], point_y=z_real[i], point_z=y_real[i], camera_x=xt, camera_y=zt, camera_z=yt, theta_x=thpitch, theta_z=0, theta_y=0)
    

    xy=camera_to_plane(XYZ,focal=f,resolution_x=U, resolution_y=V, sensor_x=s_w, sensor_y=s_h)
  

    u,v= plane_to_pixel(xy)
    point = ((u), (v))
    print("u e v sono:")
    print(point)


    cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # Cerchi rossi

    print('-----------------------------------------------')



# Salva e visualizza il risultato
output_file = ".output_image.png"
cv2.imwrite(output_file, image)

cv2.imshow("Projected Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()