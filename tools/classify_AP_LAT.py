'''
Date: 2023-04-21 10:51:19
LastEditors: zhangjian zhangjian@cecinvestment.com
LastEditTime: 2023-05-18 14:15:50
FilePath: /QC-wrist/tools/classify_AP_LAT.py
Description: 
'''
import os
import shutil
import pydicom

def classify(dst):
    cases_dir = os.listdir(dst)
    for c in cases_dir:
        cases_path = os.path.join(dst, c)
        dcms_path = os.listdir(cases_path)
        for d in dcms_path:
            dcm_file = os.path.join(cases_path, d)
            df = pydicom.read_file(dcm_file, force=True)
            ProtocolName = df.data_element('ProtocolName').value
            AccessionNumber = df.data_element(('AccessionNumber')).value
            name = AccessionNumber+'.dcm'
            if ProtocolName == '腕关节正位':
                shutil.copy(dcm_file, os.path.join('../wrist_data_dcm/wrist_AP', name))
            elif ProtocolName == '腕关节侧位':
                shutil.copy(dcm_file, os.path.join('../wrist_data_dcm/wrist_LAT', name))


if __name__ == '__main__':
    classify('../files_from_hos')