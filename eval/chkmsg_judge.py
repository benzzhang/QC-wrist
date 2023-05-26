'''
Date: 2023-05-26 16:10:41
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-26 16:50:56
FilePath: /QC-wrist/eval/chkmsg_judge.py
Description: 
'''

import math
import numpy as np

'''
    4分, 受检者基本信息是否完整? 包含ID号(唯一性)、姓名、性别、出生日期、年龄
    4分, 检查基本信息是否完整? 包含检查日期、检查时间、曝光次数/(图像数) 
    4分, 检查设备基本信息是否完整? 包含执行医院名称、设备生产商、设备型号、软件版本号
    4分, 执行参数基本信息是否完整? 包含曝光参数(KV\MA\t)、距离、电离室
    4分, 图像显示参数是否完整? 包含放大标尺、窗宽、窗位
'''
def basic_information_completed(dcmfile):
   
    completed1 = [
                dcmfile.data_element('AccessionNumber').value != None,
                dcmfile.data_element('PatientName').value != None,
                dcmfile.data_element('PatientSex').value != None,
                dcmfile.data_element('PatientBirthDate').value != None,
                dcmfile.data_element('PatientAge').value != None,]
    completed2 = [
                dcmfile.data_element('StudyDate').value != None,
                dcmfile.data_element('StudyTime').value != None,
                dcmfile.data_element('曝光次数/(图像数)').value != None,]
    completed3 = [
                dcmfile.data_element('InstitutionName').value != None,
                dcmfile.data_element('Manufacturer').value != None,
                dcmfile.data_element('ManufacturerModelName').value != None,
                dcmfile.data_element('SoftwareVersions').value != None,]
    completed4 = [
                dcmfile.data_element('KVP').value != None,
                dcmfile.data_element('DistanceSourceToDetector').value != None,
                dcmfile.data_element('电离室').value != None,]
    completed5 = [
                dcmfile.data_element('PixelSpacing').value != None,
                dcmfile.data_element('WindowWidth').value != None,
                dcmfile.data_element('WindowCenter').value != None]
    
    score = 0
    for c in [completed1, completed2, completed3, completed4, completed5]:
        if False not in c:
            score += 4
            
    return score

'''
    尺桡骨茎突连线与图像纵轴垂直, 角度90°±5以内°
'''
def dose(dcmfile):
    
    dose_value = dcmfile.data_element('DoseValue').value

    if dose_value <= 5.0:
        return True
    else:
        return False

