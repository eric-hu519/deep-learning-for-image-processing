import math
def cal_pt(points):
    if points[0][0] < points[1][0]:
        sc_x = points[1][0]
        sc_y = points[1][1]
        
        s1_x = points[0][0]
        s1_y = points[0][1]
    else:
        sc_x = points[0][0]
        sc_y = points[0][1]

        s1_x = points[1][0]
        s1_y = points[1][1]

    fh1_x = points[2][0]
    fh1_y = points[2][1]

    fh2_x = points[3][0]
    fh2_y = points[3][1]

    fh_mid = [(fh1_x + fh2_x) / 2, (fh1_y + fh2_y) / 2]
    
    s_slope = (s1_y - sc_y) / (s1_x - sc_x)
    ss_angle = abs(math.atan(s_slope) * 180 / math.pi)
    
    p_slope = (sc_y - fh_mid[1]) / (sc_x - fh_mid[0])
    p_angle = abs(math.atan(p_slope) * 180 / math.pi)
    pt_angle = 90 - p_angle
    pi_angle = pt_angle + ss_angle

    return ss_angle,pt_angle,pi_angle