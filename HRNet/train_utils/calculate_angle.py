import math
import numpy as np
#x,y is a list
def cal_pt(x,y):
    if x[0] < x[1]:
        sc_x = x[0]
        sc_y = y[0]
        
        s1_x = x[1]
        s1_y = y[1]
    else:
        sc_x = x[1]
        sc_y = y[1]

        s1_x = x[0]
        s1_y = y[0]

    fh1_x = x[2]
    fh1_y = y[2]

    fh2_x = x[3]
    fh2_y = y[3]

    fh_mid = [(fh1_x + fh2_x) / 2, (fh1_y + fh2_y) / 2]
    
    s_slope = (s1_y - sc_y) / (s1_x - sc_x)
    ss_angle = abs(math.atan(s_slope) * 180 / math.pi)
    
    p_slope = (sc_y - fh_mid[1]) / (sc_x - fh_mid[0])
    p_angle = abs(math.atan(p_slope) * 180 / math.pi)
    pt_angle = 90 - p_angle
    pi_angle = pt_angle + ss_angle
    
    return np.array([ss_angle,pt_angle,pi_angle])