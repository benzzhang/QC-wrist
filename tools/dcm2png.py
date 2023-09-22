'''
@Author     : Jian Zhang
@Init Date  : 2023-04-14 14:44
@File       : dcm2png.py
@IDE        : PyCharm
@Description: 
'''
import os
import pydicom
import cv2

def transfer(dst):
    cases_dir = os.listdir(dst)
    for c in cases_dir:
        cases_path = os.path.join(dst, c)
        dcms_path = os.listdir(cases_path)
        for d in dcms_path:
            dcm_file = os.path.join(cases_path, d)
            df = pydicom.read_file(dcm_file, force=True)

            # 'Implicit VR Little Endian' - 未压缩
            if not hasattr(df.file_meta, 'TransferSyntaxUID'):
                # DICOM defines a default Transfer Syntax, the DICOM Implicit VR Little Endian Transfer Syntax (UID = "1.2.840.10008.1.2 "),
                # which shall be supported by every conformant DICOM Implementation.
                df.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            df_pixel = df.pixel_array

            scaled_df_pixel = (df_pixel - min(df_pixel.flatten())) / (max(df_pixel.flatten()) - min(df_pixel.flatten()))

            format = '.png'
            cv2.imwrite(os.path.join('../data', c, d.replace('.dcm', format)), scaled_df_pixel*255)


if __name__ == '__main__':
    transfer('../wrist_data_dcm')