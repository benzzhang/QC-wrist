import cv2
import numpy as np
# 创建一个空白图像
image = np.zeros((400, 600, 3), dtype=np.uint8)

# 绘制双向箭头直线
pt1 = (50, 200)  # 直线的起始点坐标
pt2 = (550, 200)  # 直线的结束点坐标
color = (0, 255, 0)  # 绿色
thickness = 2
line_type = cv2.LINE_AA
shift = 0

cv2.arrowedLine(image, pt1, pt2, color, thickness, line_type, shift)

# 显示图像
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
