'''
@Author     : Jian Zhang
@Init Date  : 2023-10-17 13:41
@File       : compare_manual_model_LAT.py
@IDE        : PyCharm
@Description: 
'''
import os
import shutil

import numpy as np
import openpyxl
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics


def level_total(value):
    level = {
        'D': 0,
        'C': 1,
        'B': 2,
        'A': 3,
    }
    # 0~10~20~30~40~50~60~70~80~90~100
    levels = ['D', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'B', 'A', 'A']
    ai_score_level = level[levels[int(value / 10)]]
    return ai_score_level


def level_12(value):
    level = {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 2,
        '6': 2,
        '7': 2,
        '8': 2,
        '9': 3,
        '10': 3,
        '11': 3,
        '12': 3,
    }
    return level[str(value)]


def main(model_eval, manual_eval):
    with open(model_eval, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    wb = openpyxl.load_workbook(manual_eval)
    sheets = wb.worksheets
    sheet = sheets[0]

    rows = []
    for row in sheet.iter_rows(values_only=True, min_row=2):
        if row[0] is not None:
            cells = []
            for cell in row:
                if cell != None:
                    cells.append(cell)
            rows.append(cells)

    flag2s = []
    flag3s = []
    flag4s = []

    manual_scores = []
    model_scores = []

    manual_artifact = []
    model_artifact = []

    manual_scaphoid_is_center = []
    model_scaphoid_is_center = []

    manual_long_axis_line = []
    model_long_axis_line = []

    manual_distal_overlap = []
    model_distal_overlap = []

    for row in rows:
        tag = 'details of ' + row[0] + '.dcm'
        details = config[tag]

        # variable of list for drawing CM
        manual_score = row[5] + details['基本信息完整度']
        if details['左右标识'] == False:
            manual_score = 0
        else:
            manual_score += 20
        model_score = config[row[0] + '.dcm']
        manual_scores.append(level_total(manual_score))
        model_scores.append(level_total(model_score))

        flag1 = False if row[1] == 10 else True
        manual_artifact.append(flag1)
        model_artifact.append(details['异物伪影'])

        score = details['舟骨位于图像正中']['score']
        flag2_0 = level_12(row[2])
        flag2_1 = level_12(score)
        manual_scaphoid_is_center.append(flag2_0)
        model_scaphoid_is_center.append(flag2_1)

        score = details['腕关节长轴与影像纵轴平行']['score']
        flag3_0 = level_12(row[3])
        flag3_1 = level_12(score)
        manual_long_axis_line.append(flag3_0)
        model_long_axis_line.append(flag3_1)

        score = details['尺桡骨远端重叠']['score']
        flag4_0 = level_12(row[4])
        flag4_1 = level_12(score)
        manual_distal_overlap.append(flag4_0)
        model_distal_overlap.append(flag4_1)

        flag2s.append(abs(details['舟骨位于图像正中']['score'] - row[2]))
        flag3s.append(abs(details['腕关节长轴与影像纵轴平行']['score'] - row[3]))
        flag4s.append(abs(details['尺桡骨远端重叠']['score'] - row[4]))

    print('%.2f ± %.2f' % (np.mean(flag2s), np.std(flag2s)))
    print('%.2f ± %.2f' % (np.mean(flag3s), np.std(flag3s)))
    print('%.2f ± %.2f' % (np.mean(flag4s), np.std(flag4s)))

    # Used to filter various types of images in the confusion matrix
    for idx, (i, j) in enumerate(zip(model_scores, manual_scores)):
        if i == 0 and j == 1:
            print(rows[idx][0])
            shutil.copy(os.path.join('/data/experiments/wrist-landmark/LAT/visualized_results', rows[idx][0] + '.png'),
                        os.path.join('/data/temp', rows[idx][0] + '.png'))

    cnt = 0
    for i, j in zip(manual_scores, model_scores):
        if i == j:
            cnt += 1
    print('total Acc:', cnt / len(manual_scores))

    cnt = 0
    for i, j in zip(manual_scores, model_scores):
        if j == 3 and i == j:
            cnt += 1
    print('A-Precision:', cnt / model_scores.count(3))
    print('A-Recall:', cnt / manual_scores.count(3))

    cnt = 0
    for i, j in zip(manual_scores, model_scores):
        if j == 2 and i == j:
            cnt += 1
    print('B-Precision:', cnt / model_scores.count(2))
    print('B-Recall:', cnt / manual_scores.count(2))

    cnt = 0
    for i, j in zip(manual_scores, model_scores):
        if j == 1 and i == j:
            cnt += 1
    print('C-Precision:', cnt / model_scores.count(1))
    print('C-Recall:', cnt / manual_scores.count(1))

    cnt = 0
    for i, j in zip(manual_scores, model_scores):
        if j == 0 and i == j:
            cnt += 1
    print('D-Precision:', cnt / model_scores.count(0))
    print('D-Recall:', cnt / manual_scores.count(0))

    # config of DRAW
    font1 = {'size': 15}
    title_font_size = 18
    annot_kws = {'size': 13, 'weight': 'bold'}
    labelsize = 13
    color = 'Reds'

    # draw confusion_matrix of scores
    CM_scores = metrics.confusion_matrix(model_scores, manual_scores, labels=[0, 1, 2, 3])
    labels = ['D', 'C', 'B', 'A']
    hm = sns.heatmap(CM_scores, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Evaluation Level', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/LAT-scores.png')

    # draw ~ of artifact
    plt.clf()
    CM_artifact = metrics.confusion_matrix(model_artifact, manual_artifact)
    labels = ['False', 'True']
    hm = sns.heatmap(CM_artifact, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Artifact', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/LAT-artifact.png')

    # draw ~ of scaphoid
    plt.clf()
    CM_scaphoid = metrics.confusion_matrix(model_scaphoid_is_center, manual_scaphoid_is_center, labels=[0, 1, 2, 3])
    labels = ['Fail', 'Poor', 'Average', 'Excellent']
    hm = sns.heatmap(CM_scaphoid, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Scaphoid', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/LAT-scaphoid.png')

    # draw ~ of long axis line
    plt.clf()
    CM_long_axis_line = metrics.confusion_matrix(model_long_axis_line, manual_long_axis_line, labels=[0, 1, 2, 3])
    hm = sns.heatmap(CM_long_axis_line, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels,
                     yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Long Axis Line', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/LAT-long_axis_line.png')

    # draw ~ of distal overlap
    plt.clf()
    CM_distal_overlap = metrics.confusion_matrix(model_distal_overlap, manual_distal_overlap, labels=[0, 1, 2, 3])
    hm = sns.heatmap(CM_distal_overlap, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels,
                     yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of \nthe Distal Overlap', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/LAT-distal_overlap.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='compare manual eval with model eval')
    parser.add_argument('--model-eval', type=str, default='../inference_result/inference_LAT.json')
    parser.add_argument('--manual-eval', type=str, default='./manual_evaluation_LAT.xlsx')
    args = parser.parse_args()
    main(args.model_eval, args.manual_eval)
