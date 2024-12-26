import numpy as np
import matplotlib.pyplot as plt
import cv2


# Matrice di rotazione attorno all'asse Y

#matrice di yaw
def R_z(theta_z):
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    return Rz

#matrice di pitch
def R_y(theta_y):
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    return Ry

#matrice di roll
def R_x(theta_x):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    return Rx


#x,y,z è la posizione della camera rispetto al mondo
#dato un punto X,Y,Z del mondo lo trasforma in X-x, Y-y, Z-z della telecamera
def C(x,y,z):
    C = np.array([
        [1,0,0,-x],
        [0,1,0,-y],
        [0,0,1,-z],
        [0,0,0,1]
    ])
    return C

#punti nel mondo reale
def UVW(x,y,z):
    UVW = np.array([
        [x],
        [y],
        [z],
        [1]
    ])
    return UVW

def rotation(pitch,roll,yaw):
    R = np.array([
        [pitch],
        [roll],
        [yaw]
    ])
    return R

def T(xc,yc,zc):
    T = np.array([
        [xc],
        [yc],
        [zc],
    ])
    return T

def WorldPoint(x,y,z):
    WorldPoint = np.array([
        [x, y ,z]
    ])
    return WorldPoint

def CameraMatrix(fx,fy,ox,oy):
    ff = np.array([
        [fx, 0, ox],
        [0, fy, oy],
        [0, 0, 1]
    ])
    return ff

def real_to_camera(point_x, point_y, point_z, camera_x, camera_y, camera_z,  theta_x=0, theta_y=0, theta_z=0):
    translation_matrix = C(camera_x, camera_y, camera_z)
    
    # Poi, applica la rotazione (intorno agli assi Z, Y, e X)
    rotation_matrix = R_y(theta_y) @ R_x(theta_x) @ R_z(theta_z)  # Assumiamo l'ordine di rotazione Z-Y-X
    
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



#forse basta solo f=f/sx
def focal_length(focal,s_resolution,p_sensor):
    return focal*s_resolution/p_sensor


def camera_to_plane(XYZ,focal,s_resolutionx,s_resolutiony,p_sensorx,p_sensory):

    fx = focal_length(focal, s_resolutionx, p_sensorx)  # Calcolo della focale in x
    fy = focal_length(focal, s_resolutiony, p_sensory)  # Calcolo della focale in y

    ##Vuole centro dell'immagine o centro degli assi?##
    ox = s_resolutionx / 2  # Centro dell'immagine in x
    oy = s_resolutiony / 2 # Centro dell'immagine in y, punto in cui gli assi sono (0,0)

    # Matrice di proiezione
    ff = np.array([
        [fx, 0, ox, 0],
        [0, fy, oy, 0],
        [0, 0, 1, 0]
    ])

    # Moltiplicazione della matrice di proiezione per le coordinate 3D
    xy_homogeneous = ff @ XYZ  # Risultato in coordinate omogenee (3x1)

    # Normalizzazione per ottenere le coordinate in 2D nel piano immagine
    x = xy_homogeneous[0] / xy_homogeneous[2]  # Coordinata x normalizzata
    y = xy_homogeneous[1] / xy_homogeneous[2]  # Coordinata y normalizzata

    xy = np.array([x, y])  # Coordinata omogenea normalizzata  


    print("fx:")
    print(fx)
    print("fy:")
    print(fy)
    print("ff:")
    print(ff)
    print("xy_homogeneous:")
    print(xy_homogeneous)
    print("xy:")
    print(xy)

    return xy





def plane_to_pixel(xy):
    u = xy[0]  # Coordinata u (pixel x)
    v = xy[1]  # Coordinata v (pixel y)
    return int(u[0]), int(v[0])


# Lettura immagine con OpenCV
image_file = "./imgs/test20241212.png"
image = cv2.imread(image_file)



# Dati iniziali
x_real = np.array([-2.5, 0.5, 0.5, 4.6])
y_real = np.array([13.41, 8.00, 13.00, 10.91])
z_real = np.zeros_like(x_real)

xt= 0
yt= 0
zt= 7.20  # Coordinate della camera
thyaw = 0 * np.pi / 180  # Yaw (rotazione attorno a Z)
thpitch = -32 * np.pi / 180  # Pitch (rotazione attorno a Y) (radianti)
throll = 0 * np.pi / 180  # Roll (rotazione attorno a X)
# Parametri immagine
f = 0.003  # Distanza focale (m)
s_w = 0.00498  # Larghezza sensore (m)
s_h = 0.00374  # Altezza sensore (m)
U = 1280   # Larghezza immagine (pixel)
V = 720  # Altezza immagine (pixel)



print("larghezza ed altezza immagine:")
print(U,V)


# Disegna i punti sull'immagine

for i in range(len(x_real)):
    print("point:")
    print(x_real[i],z_real[i], y_real[i])

    #invertire y con z 
    #XYZ=real_to_camera(point_x=x_real[i], point_y=y_real[i], point_z=z_real[i], camera_x=xt, camera_y=yt, camera_z=zt, theta_x=thpitch, theta_z=0, theta_y=0)
    

    #xy=camera_to_plane(XYZ,focal=f,s_resolutionx=U,s_resolutiony=V,p_sensorx=s_w,p_sensory=s_h)
  
    objectPoints = WorldPoint(x_real[i], z_real[i], y_real[i])

    rvec = rotation(thpitch,0,0)  # Converte in rvec
    tvec = T(xt,-zt,yt)
    print(tvec)
    
    dist_coeffs = np.zeros(4)
    cameraMatrix = CameraMatrix(focal_length(f,U,s_w), -focal_length(f,V,s_h), U/2, V/2)
    points, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, dist_coeffs)
    print(points[0][0])
    point = ( int(points[0][0][0]), int(points[0][0][1]))

    # u,v= plane_to_pixel(xy)
    # point = ((u), (v))
    # #point = (int(U/2), int(V/2))
    # print("u e v sono:")
    # print(point)

    cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # Cerchi rossi

    print('-----------------------------------------------')



# Salva e visualizza il risultato
output_file = "./imgs/output_image.png"
cv2.imwrite(output_file, image)

cv2.imshow("Projected Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()