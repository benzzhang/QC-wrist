#!/bin/bash
###
 # @Date: 2023-06-03 14:00:59
 # @LastEditors: zhangjian zhangjian@cecinvestment.com
 # @LastEditTime: 2023-07-27 17:37:46
 # @FilePath: /QC-wrist/train.sh
 # @Description: 
### 

python train_landmark.py --config-file 'configs/config_landmarks_AP.yaml'
python train_landmark.py --config-file 'configs/config_landmarks_LAT.yaml'