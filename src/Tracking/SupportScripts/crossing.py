# Evauate if a persco cross a line


def orientation(p, q, r):
    '''
     evaulate the orientation of the r point respect to the line formed by the p and q points
    '''
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

# Funzione per verificare se due segmenti si intersecano
def on_segment(p, q, r):
    '''
    check if the point q is in the middle of the segment formed by the p and r points
    '''
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])



def do_intersect(box, box2, p2, q2):
    '''
    box-> bounding box's top left point and bottom right point (x1, x2, y1, y2)
    box2-> for the next frame  (x1, x2, y1, y2)
    p2-> line's point 1 (x1, y1)
    q2-> line's point 2 (x2, y2)
    '''
    x1,x2,y1,y2 = box
    x3,x4,y3,y4 = box2
    
    box= ((x1+x2)/2,y2) #x2-x1/2+x1 =(x1+x2)/2 (central point of the bottom of the bounding box)
    box2= ((x3+x4)/2,y4) #x2-x1/2+x1 =(x1+x2)/2 (central point of the bottom of the bounding box)


    # Calcolare le 4 orientazioni
    o1 = orientation(box, box2, p2)
    o2 = orientation(box, box2, q2)
    o3 = orientation(p2, q2, box)
    o4 = orientation(p2, q2, box2)
    # only it evry orientation is different the two segments intersect
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    # if the orientation is 0 the points are collinear
    if o1 == 0 and on_segment(box, p2, box2): # if the point is in the middle of the segment
        return True
    if o2 == 0 and on_segment(box, q2, box2):
        return True
    if o3 == 0 and on_segment(p2, box, q2):
        return True
    if o4 == 0 and on_segment(p2, box2, q2):
        return True
    return False
