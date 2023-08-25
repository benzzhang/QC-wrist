'''
@Author     : Jian Zhang
@Init Date  : 2023-04-19 13:35
@File       : gen_list_meta.py
@IDE        : PyCharm
@Description: [c1,c2,c3]
                c1: 0-无运动伪影 , 1-有运动伪影
                c2: 0-无异物伪影 , 1-有异物伪影
                c3: 0-关节面可见 , 1-关节面不可见
'''
import json
import os
import random
import time


def random_num(len):
    train_idx = random.sample(range(0, len), int(len * 0.6))
    return train_idx


def gen_list_meta(path, ldm_name_list):
    # clear .txt
    f1 = './' + os.path.basename(path) + '_class_list.txt'
    f2 = './' + os.path.basename(path) + '_landmarks_list.txt'
    f3 = './' + os.path.basename(path) + '_list.txt'
    open(f1, 'w').close()
    open(f2, 'w').close()
    open(f3, 'w').close()

    files = os.listdir(path)
    # files.sort(key=lambda x: int(x.split('.')[0][2:]))
    cnt = len(files) // 2

    for file in files:
        if '.json' in file:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as json_file:
                json_dict = json.load(json_file)
                flags_list = []
                shapes_dict = {}

                # tag of classes
                for idx, value in enumerate(list(json_dict["flags"].values())):
                    label = int(value)
                    if idx > 2:
                        break
                    flags_list.append(label)

                # size of image
                H = json_dict["imageHeight"]
                W = json_dict["imageWidth"]

                # scale position to 0.~1.
                for ldm_name in ldm_name_list:
                    is_matched = False
                    for landmark in json_dict["shapes"]:
                        # 'landmark name'
                        markname = landmark["label"]
                        if markname == ldm_name:
                            is_matched = True
                            # 'position'
                            y = format(landmark["points"][0][1] / H, '.4f')
                            x = format(landmark["points"][0][0] / W, '.4f')
                            value = [float(y), float(x)]

                            shapes_dict[str(ldm_name)] = value
                        if not is_matched:
                            shapes_dict[str(ldm_name)] = [float(0), float(0)]

                # write 'classes' to .txt var LIST
                flags = open(f1, 'a', encoding='utf-8')
                for i in flags_list:
                    flags.write(str(i) + ' ')
                flags.write('\n')
                flags.close()

                # write 'label:position' to .txt var DICT
                shapes = open(f2, 'a', encoding='utf-8')
                for i in shapes_dict.values():
                    for j in i:
                        shapes.write(str(j) + ' ')
                shapes.write('\n')
                shapes.close()

            # write 'file name' to .txt
            image_list = open(f3, 'a')
            image_list.write(file.replace('.json', '.png') + '\n')
            image_list.close()


    # random split into TRAIN & TEST
    train_idx = random_num(cnt)

    with open(f1, 'r') as f:
        train, test = [], []
        lines = [l.strip() for l in f.readlines()]
        for i in range(cnt):
            if i in train_idx:
                train.append(lines[i])
            else:
                test.append(lines[i])

        ftrain = open(f1.replace('_class', '_train_class'), 'w', encoding='utf-8')
        for i in train:
            ftrain.write(i + '\n')
        ftrain.close()
        ftest = open(f1.replace('_class', '_valid_class'), 'w', encoding='utf-8')
        for i in test:
            ftest.write(i + '\n')
        ftest.close()

    with open(f2, 'r') as f:
        train, test = [], []
        lines = [l.strip() for l in f.readlines()]
        for i in range(cnt):
            if i in train_idx:
                train.append(lines[i])
            else:
                test.append(lines[i])

        ftrain = open(f2.replace('_landmarks', '_train_landmarks'), 'w', encoding='utf-8')
        for i in train:
            ftrain.write(i + '\n')
        ftrain.close()
        ftest = open(f2.replace('_landmarks', '_valid_landmarks'), 'w', encoding='utf-8')
        for i in test:
            ftest.write(i + '\n')
        ftest.close()

    with open(f3, 'r') as f:
        train, test = [], []
        lines = [l.strip() for l in f.readlines()]
        for i in range(cnt):
            if i in train_idx:
                train.append(lines[i])
            else:
                test.append(lines[i])

        ftrain = open(f3.replace('_list', '_train_list'), 'w', encoding='utf-8')
        for i in train:
            ftrain.write(i + '\n')
        ftrain.close()
        ftest = open(f3.replace('_list', '_valid_list'), 'w', encoding='utf-8')
        for i in test:
            ftest.write(i + '\n')
        ftest.close()


if __name__ == '__main__':
    ldm_name_list_AP = ['P1-拇指指掌关节', 'P2-桡骨茎突', 'P3-尺骨茎突']
    ldm_name_list_LAT = ['P1-舟骨中心', 'P2-桡骨远端中心', 'P3-桡骨近端中心', 'P4-尺骨远端中心', 'P5-尺骨近端中心']
    
    '''
        from LISTS include AP/LAT for landmarks
        after executing, delete LISTS about classifying
    '''
    # gen_list_meta('./wrist_AP', ldm_name_list_AP)
    # gen_list_meta('./wrist_LAT', ldm_name_list_LAT)

    '''
        from LISTS include AP/LAT for classifying
        after executing, delete LISTS about landmarks
    '''
    gen_list_meta('./wrist', ldm_name_list_AP)