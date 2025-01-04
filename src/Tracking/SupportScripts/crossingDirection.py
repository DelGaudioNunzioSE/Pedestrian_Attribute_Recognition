#Afther crossig we evaluate the direction of crossing
import numpy as np


def calculate_crossing(box, box2, center, arrowEnd):
    '''
        box -> bounding box's top left point and bottom right point (x1, x2, y1, y2)
        box2 -> for the next frame  (x1, x2, y1, y2)
        center -> center of the line (x, y)
        arrowEnd -> point where the perpendicular arrow ends (orientation line) (x, y)
    '''
    vet_track = np.array([box2[1] - box[1], box2[3] - box[3]]) #vector module and direction (x2,y2)
    vet_line = np.array([arrowEnd[0] - center[0], arrowEnd[1] - center[1]]) #vector module and direction
    dot_product = np.dot(vet_track, vet_line) # dot product between the two vectors
    if dot_product > 0:
        return True #if the dot product is positive the vectors are in the same direction
    else: 
        return False