'''
@Author     : Jian Zhang
@Init Date  : 2023-04-18 16:33
@File       : medical_augment.py
@IDE        : PyCharm
@Description: 
'''
import torchvision.transforms as transforms
import albumentations as A
def XrayTrainTransform(img_size=256, crop_size=224):
   return A.Compose([
       # A.Resize(img_size, img_size),
       A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=90),
       A.VerticalFlip(p=0.5),
       A.HorizontalFlip(p=0.5),
       A.RandomBrightnessContrast(p=0.2),
   ])

#if __name__ == "__main__":
#    import cv2
#    img = cv2.imread('baks/1.jpg')
#    transforms = XrayTrainTransform()
#    new_img = transforms(image=img)['image']
#    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
#    cv2.imshow('new_img', new_img)
#    key=cv2.waitKey(-1)

