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
import os
import random


def random_num(len):
    train_idx = random.sample(range(0, len), int(len * 0.6))
    return train_idx


def gen_list_meta(path):
    # clear .txt
    f1 = './' + os.path.basename(path) + '_position_class_list.txt'
    f3 = './' + os.path.basename(path) + '_position_list.txt'
    open(f1, 'w').close()
    open(f3, 'w').close()

    files_AP = [i for i in os.listdir(path) if 'AP' in i]
    files_LAT = [i for i in os.listdir(path) if 'LAT' in i]

    # files.sort(key=lambda x: int(x.split('.')[0][2:]))
    cnt_AP = len(files_AP)
    cnt_LAT = len(files_LAT)

    flags = open(f1, 'a', encoding='utf-8')
    for file in files_AP:
        flags.write('1')
        flags.write('\n')
    for file in files_LAT:
        flags.write('0')
        flags.write('\n')
    flags.close()

    names = open(f3, 'a', encoding='utf-8')
    for file in files_AP:
        names.write(str(file))
        names.write('\n')
    for file in files_LAT:
        names.write(str(file))
        names.write('\n')
    names.close()

    # random split into TRAIN & TEST
    train_idx = random_num(cnt_AP+cnt_LAT)

    with open(f1, 'r') as f:
        train, test = [], []
        lines = [l.strip() for l in f.readlines()]
        for i in range(cnt_AP+cnt_LAT):
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

    with open(f3, 'r') as f:
        train, test = [], []
        lines = [l.strip() for l in f.readlines()]
        for i in range(cnt_AP+cnt_LAT):
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
    '''
        from LISTS include AP&LAT for classifying
    '''
    gen_list_meta('./wrist')