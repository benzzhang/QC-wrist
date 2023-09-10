'''
Date: 2023-05-26 16:10:41
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-09-05 13:56:15
FilePath: /QC-wrist/eval/chkmsg_judge.py
Description: 
    4分, 受检者基本信息是否完整? 包含ID号(唯一性)、姓名、性别、出生日期、年龄
    4分, 检查基本信息是否完整? 包含检查日期、检查时间
    4分, 检查设备基本信息是否完整? 包含执行医院名称、设备生产商、设备型号、软件版本号
    4分, 执行参数基本信息是否完整?包含管电压、距离、曝光时间
    4分, 图像显示参数是否完整? 包含放大标尺、窗宽、窗位
'''
def basic_information_completed(dcmfile):

    standard_tags_dr_wrist = [
        'StudyID', 'PatientName', 'PatientSex', 'PatientBirthDate', 'PatientAge',
        'StudyDate', 'StudyTime',
        'InstitutionName', 'Manufacturer', 'ManufacturerModelName', 'SoftwareVersions',
        'KVP', 'DistanceSourceToDetector', 'ExposureTime',
        'PixelSpacing', 'WindowWidth', 'WindowCenter'
    ]

    all_dcm_tags = dcmfile.dir()
    score = 0
    for std in standard_tags_dr_wrist:
        if std in all_dcm_tags and dcmfile.data_element(std).value != None:
            score += 2
    
    return score

'''
    辐射剂量(每次摄影不高于5mGy)
'''
def dose(dcmfile):
    try:
        dose_value = dcmfile.data_element('DoseValue').value
        if dose_value <= 5.0:
            return True
        else:
            return False
    except:
        return False


