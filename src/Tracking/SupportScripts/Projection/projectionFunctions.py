import numpy as np
import matplotlib.pyplot as plt
import cv2


# Matrice di rotazione attorno all'asse Y

def R_z(theta_z):
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0,0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return Rz


def R_y(theta_y):
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y),0],
        [0, 1, 0, 0],
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

def RotationMatrix(theta_x, theta_y, theta_z):
    R = R_z(theta_z) @ R_x(theta_x) @ R_y(theta_y) #zxy, yaw pitch and roll
    return R

def C(x,y,z):
    C = np.array([
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [0, 0, 0, 1]
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

def transformationMatrix(rotation_matrix, translation_matrix):
    transformation_matrix = rotation_matrix @ translation_matrix
    return transformation_matrix


def real_to_camera(point_x, point_y, point_z, camera_x, camera_y, camera_z,  theta_x=0, theta_y=0, theta_z=0):
    translation_matrix = C(camera_x, camera_y, camera_z)
    
    # Then, apply the rotation (around the Z, Y, and X axes)
    rotation_matrix = RotationMatrix(theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)  # Let's assume the rotation order Z-Y-X
    
    # Now let's combine the transformation (rotation + translation)
    transformation_matrix = transformationMatrix(rotation_matrix=rotation_matrix, translation_matrix=translation_matrix)

    # We apply the transformation to the point in UVW coordinates
    UVW_point = UVW(x=point_x, y=point_y, z=point_z)
    XYZ = transformation_matrix @ UVW_point

    return XYZ




def focal_length(f,resolution,sensor):
    s= sensor/resolution
    return f/s


def camera_to_plane(XYZ,focal, resolution_x, resolution_y, sensor_x, sensor_y):

    fx = focal_length(f=focal, resolution=resolution_x, sensor=sensor_x)  # Calculating the focal length in x
    fy = -focal_length(f=focal, resolution=resolution_y, sensor=sensor_y)  # Calculating the focal length in y

    ox = resolution_x / 2  # Center of the image in x
    oy = resolution_y / 2  # Center of the image in y

    # We asume that the pixel are square
    perspective_projection = np.array([
        [fx, 0, ox, 0],
        [0, -fx, oy, 0],
        [0, 0, 1, 0]
    ])

    # Moltiplicazione della matrice di proiezione per le coordinate 3D
    uvw = perspective_projection @ XYZ  # Result in homogeneous coordinates (3x1)

    # Normalization to obtain 2D coordinates in the image plane
    x = uvw[0] / uvw[2]  # Normalized x-coordinate
    y = uvw[1] / uvw[2]  # Normalized y-coordinate

    xy = np.array([x, y])  # Normalized homogeneous coordinate  

    return xy





def plane_to_pixel(xy):
    u = xy[0]  # u-coordinate (pixel x)
    v = xy[1]  # v-coordinate (y-pixel)
    return int(u[0]), int(v[0])




# Disegna i punti sull'immagine

def draw_points(image, x_real, y_real, z_real, camera_x, camera_y, camera_z, thyaw, thpitch, throll, focal, resolution_x, resolution_y, sensor_x, sensor_y):


    for i in range(len(x_real)):


        XYZ=real_to_camera(point_x=x_real[i], point_y=y_real[i], point_z=z_real[i], camera_x=camera_x, camera_y=camera_y, camera_z=camera_z, theta_x=thpitch, theta_y=throll, theta_z=thyaw)
        

        xy=camera_to_plane(XYZ,focal=focal,resolution_x=resolution_x, resolution_y=resolution_y, sensor_x=sensor_x, sensor_y=sensor_y)
    

        u,v= plane_to_pixel(xy)
        point = ((u), (v))


        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # red circle (points)

        print('-----------------------------------------------')

    return image




def points(x_real, y_real, z_real, camera_x, camera_y, camera_z, thyaw, thpitch, throll, focal, resolution_x, resolution_y, sensor_x, sensor_y):


    points=[]
    for i in range(len(x_real)):


        XYZ=real_to_camera(point_x=x_real[i], point_y=y_real[i], point_z=z_real[i], camera_x=camera_x, camera_y=camera_y, camera_z=camera_z, theta_x=thpitch, theta_y=throll, theta_z=thyaw)
        

        xy=camera_to_plane(XYZ,focal=focal,resolution_x=resolution_x, resolution_y=resolution_y, sensor_x=sensor_x, sensor_y=sensor_y)
    

        u,v= plane_to_pixel(xy)
        point = ((u), (v))

        points.append(point)


    return points


def inversion_draw_points(image, x_real, y_real, z_real, camera_x, camera_y, camera_z, thpitch, throll, thyaw, focal, resolution_x, resolution_y, sensor_x, sensor_y):

    return draw_points(image=image, x_real=x_real, y_real=z_real, z_real=y_real, camera_x=camera_x, camera_y=camera_z, camera_z=camera_y, thyaw=throll, thpitch=thpitch, throll=thyaw, focal=focal, resolution_x=resolution_x, resolution_y=resolution_y, sensor_x=sensor_x, sensor_y=sensor_y)

def inversion_points( x_real, y_real, z_real, camera_x, camera_y, camera_z, thpitch, throll, thyaw, focal, resolution_x, resolution_y, sensor_x, sensor_y):

    return  points(x_real=x_real, y_real=z_real, z_real=y_real, camera_x=camera_x, camera_y=camera_z, camera_z=camera_y, thyaw=throll, thpitch=thpitch, throll=thyaw, focal=focal, resolution_x=resolution_x, resolution_y=resolution_y, sensor_x=sensor_x, sensor_y=sensor_y)