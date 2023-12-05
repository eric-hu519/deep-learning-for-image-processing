import math
import numpy as np
import matplotlib.pyplot as plt

def compute_angles(keypoints):
    a = keypoints[:2]
    b = keypoints[2:4]
    c = keypoints[4:6]
    d = keypoints[6:]
    
    # 保证c点高于d点
    if c[1] >= d[1]:
        c, d = d, c
    
    # 计算c,d点的中点
    mid = [(c[0]+d[0])/2, (c[1]+d[1])/2]
    # 计算过mid点与b点的直线与过mid点的垂线形成的夹角
    pt = math.atan2(b[1] - mid[1], b[0] - mid[0])
    # 计算过mid点与b点的直线与过b点且垂直于a点与b点的直线形成的夹角
    pi = math.atan2(a[1] - b[1], a[0] - b[0])
    # 计算过a点与b点的直线与水平线形成的夹角
    ss = math.atan2(b[1] - a[1], b[0] - a[0])

    angle = [pt, pi, ss]
    
    return angle



def plot_keypoints(keypoints, angle):
    a = keypoints[:2]
    b = keypoints[2:4]
    c = keypoints[4:6]
    d = keypoints[6:]
    
    # 保证c点高于d点
    if c[1] >= d[1]:
        c, d = d, c
    
    # 计算c,d点的中点
    mid = [(c[0]+d[0])/2, (c[1]+d[1])/2]
    
    # 绘制keypoints
    plt.scatter(keypoints[0], keypoints[1], color='red', label='a')
    plt.scatter(keypoints[2], keypoints[3], color='blue', label='b')
    plt.scatter(keypoints[4], keypoints[5], color='green', label='c')
    plt.scatter(keypoints[6], keypoints[7], color='orange', label='d')
    
    # 绘制直线
    plt.plot([a[0], b[0]], [a[1], b[1]], color='black', label='a-b')
    plt.plot([c[0], d[0]], [c[1], d[1]], color='gray', label='c-d')
    
    # 绘制mid点
    plt.scatter(mid[0], mid[1], color='purple', label='mid')
    
    # 绘制夹角
    plt.plot([mid[0], b[0]], [mid[1], b[1]], linestyle='dashed', color='magenta')
    plt.plot([mid[0], mid[0]+np.cos(angle[0])], [mid[1], mid[1]+np.sin(angle[0])], linestyle='dashed', color='cyan')
    plt.plot([b[0], b[0]+np.cos(angle[1])], [b[1], b[1]+np.sin(angle[1])], linestyle='dashed', color='yellow')
    plt.plot([a[0], b[0]], [a[1], b[1]], linestyle='dashed', color='black')
    
    # 绘制水平线
    plt.plot([a[0], b[0]], [a[1], a[1]], linestyle='dashed', color='gray')
    #绘制过mid点垂线
    plt.plot([mid[0], mid[0]], [mid[1], b[1]], linestyle='dashed', color='gray')
    #将ab连线绕b点顺时针旋转90度，以虚线作出该线段
    plt.plot([b[0], b[0]+np.sin(angle[2])], [b[1], b[1]+np.cos(angle[2])], linestyle='dashed', color='black')
    # 标注夹角度数
    plt.text(mid[0]+0.5, mid[1]-0.5, f'{np.degrees(angle[0]):.2f}°', color='magenta')
    plt.text(mid[0]+0.5, mid[1]+0.5, f'{np.degrees(angle[1]):.2f}°', color='yellow')
    plt.text(a[0]+0.5, a[1]-0.5, f'{np.degrees(angle[2]):.2f}°', color='black')
    
    # 标注水平线夹角度数
    
    # 设置坐标轴范围
    plt.xlim(0, max(keypoints) + 5)
    plt.ylim(0, max(keypoints) + 5)
    
    # 设置坐标轴标签
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 显示图例
    plt.legend()
    
    # 显示图形
    plt.gca().invert_yaxis()  # 反转y轴
    plt.show()

# 示例数据
keypoints = [24.6, 40.8, 30.9, 34.4, 22.6, 54.8, 15.1, 68.5]
angle = compute_angles(keypoints)

# 绘制图形
plot_keypoints(keypoints, angle)
print(angle)
