#!/bin/bash

python train_landmark.py --config-file 'experiments/config_landmarks_AP.yaml'
python train_landmark.py --config-file 'experiments/config_landmarks_LAT.yaml'