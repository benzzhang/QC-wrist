'''
Date: 2023-05-24 14:14:31
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-26 11:40:51
FilePath: /QC-wrist/tools/form_matched_image.py
Description: 
'''

if __name__ == '__main__':
        
    import cv2

    # 事先准备L、R字符图片, 对其进行裁剪, 保存作为标准对照图
    file = 'R-pre.png' # file = 'L-pre.png'
    char = cv2.imread(filename=file, flags=cv2.IMREAD_GRAYSCALE)
    str_x, str_y = -1, -1
    end_x, end_y = -1, -1
    for i in range(char.shape[0]):
        for j in range(char.shape[1]):
            if char[i][j] != 0:
                if str_x == -1 and str_y == -1:
                    str_x, str_y = i, j
                end_x, end_y = i, j
    charr = char[str_x:end_x, str_y:end_y]
    cv2.imwrite('char-RR.png', charr)

    # 缩放到固定尺, 二值化, 保存
    cv2.imwrite('char-R.png', cv2.threshold(cv2.resize(cv2.imread('char-RR.png.png', cv2.IMREAD_GRAYSCALE), (64,64)),0,255,cv2.THRESH_BINARY)[1])
    cv2.imwrite('char-L.png', cv2.threshold(cv2.resize(cv2.imread('char-LL.png', cv2.IMREAD_GRAYSCALE), (64,64)),0,255,cv2.THRESH_BINARY)[1])