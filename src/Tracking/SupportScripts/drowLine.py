import cv2
import numpy as np

line_dict={
    "line":{}
}

def drawLine(frame,p1,p2, i):
    '''
    frame-> where to draw the line
    p1-> point 1 of the line
    p2-> point 2 of the line
    i-> id line
    '''
    # drow the line
    cv2.line(frame, p1,p2, color=(255, 0, 0), thickness=3) 
    cx = (p1[0] + p2[0]) // 2
    cy = (p1[1] + p2[1]) // 2
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    #dorw the points of the line
    cv2.circle(frame, p1 , 3,(0, 0, 255), thickness=3)
    cv2.circle(frame, p2 , 3,(0, 0, 255), thickness=3)

    length = np.sqrt(dx**2 + dy**2) 
    unit_dx = dx / length
    unit_dy = dy / length

    perp_dx = -unit_dy
    perp_dy = unit_dx
    arrowEnd=(int(cx+perp_dx*25),int(cy+perp_dy*25))
    cv2.arrowedLine(frame, (cx, cy), arrowEnd, (255, 0, 0), thickness=3)
    if(p1[0]<p2[0]):
        cv2.putText(frame, str(i),(p1[0],p1[1]-25),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
    else:
        cv2.putText(frame, str(i),(p2[0],p2[1]-25),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
    new_line = {        "id": i,
                        "p1": p1,
                        "p2": p2,
                        "center": (cx,cy),
                        "arrowEnd": arrowEnd
                    }
    line_dict["line"][new_line["id"]]= new_line
    return frame