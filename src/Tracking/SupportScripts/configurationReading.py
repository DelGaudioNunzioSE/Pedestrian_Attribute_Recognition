# Create the dictionary config with the configuration parameters

import json

import numpy as np

from Projection.projectionFunctions import inversion_points

config={
    "x_real":[], #we have to inizialize in order to append the points
    "y_real":[]
}

def get_config(file_path):
    '''Read the configuration file and return the configuration dictionary'''
    
    with open(file_path, 'r') as f:
        f_config = json.load(f)
        lines=f_config["lines"]                     #Per disegnare linee 
        for line in f_config["lines"]:              #Per proiettare punti 
            config["x_real"].append(line["x1"])
            config["y_real"].append(line["y1"])
            config["x_real"].append(line["x2"])
            config["y_real"].append(line["y2"])

        config["z_real"] = np.zeros_like(config["x_real"])
        config["xc"] = f_config["xc"]
        config["yc"] = f_config["yc"]
        config["zc"] = f_config["zc"]
        config["thyaw"] = f_config["thyaw"] 
        config["thpitch"] = f_config["thpitch"]
        config["throll"] = f_config["throll"] 
        config["U"] = f_config["U"]  # Larghezza immagine (pixel)
        config["V"] = f_config["V"]  # Altezza immagine (pixel)
        config["f"] = f_config["f"]
        config["s_w"] = f_config["sw"]
        config["s_h"] = f_config["sh"]
        print(config)
    return config, lines
           
def getPoints(config_path='./src/config/config.txt'):
    '''Create the points's list for the projection''' 
    config,lines = get_config(config_path)

    return inversion_points(
        x_real=config["x_real"],
        y_real=config["y_real"],
        z_real = config["z_real"],
        camera_x=config["xc"],
        camera_y=config["yc"],
        camera_z=config["zc"],
        thyaw=config["thyaw"],
        thpitch=config["thpitch"],
        throll=config["throll"],
        focal=config["f"],
        resolution_x=config["U"],
        resolution_y=config["V"],
        sensor_x=config["s_w"],
        sensor_y=config["s_h"]
    ), lines