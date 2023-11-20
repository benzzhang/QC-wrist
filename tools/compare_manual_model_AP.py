'''
@Author     : Jian Zhang
@Init Date  : 2023-10-13 16:57
@File       : compare_manual_model_AP.py
@IDE        : PyCharm
@Description: 使用json文件及人工质控评价结果, 绘制混淆矩阵图
'''
import os.path
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


def level_10(value):
    level = {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 2,
        '5': 2,
        '6': 2,
        '7': 2,
        '8': 3,
        '9': 3,
        '10': 3
    }
    return level[str(value)]


def level_10to3(value):
    level = {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 1,
        '5': 1,
        '6': 2,
        '7': 2,
        '8': 2,
        '9': 2,
        '10': 2
    }
    return level[str(value)]


def level_5(value):
    level = {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 2,
        '4': 2,
        '5': 3
    }
    return level[str(value)]


def level_6(value):
    level = {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 2,
        '4': 2,
        '5': 3,
        '6': 3
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

    flag3s = []
    flag4s = []
    flag5s = []
    flag6s = []

    manual_scores = []
    model_scores = []

    manual_artifact = []
    model_artifact = []

    manual_joint = []
    model_joint = []

    manual_midpoint_is_center = []
    model_midpoint_is_center = []

    manual_line_is_horizontal = []
    model_line_is_horizontal = []

    manual_include_radius_ulna = []
    model_include_radius_ulna = []

    manual_distance_to_edge = []
    model_distance_to_edge = []

    for row in rows:
        tag = 'details of ' + row[0] + '.dcm'
        details = config[tag]

        # variable of list for drawing CM
        manual_score = row[7] + details['基本信息完整度']
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

        flag2_0 = True if row[2] == 5 else False
        flag2_1 = True if details['上缘包含拇指指掌关节'] == 5 else False
        manual_joint.append(flag2_0)
        model_joint.append(flag2_1)

        score = details['尺桡骨茎突连线中点位于图像正中']['score']
        flag3_0 = level_10to3(row[3])
        flag3_1 = level_10to3(score)
        manual_midpoint_is_center.append(flag3_0)
        model_midpoint_is_center.append(flag3_1)

        score = details['尺桡骨茎突连线与影像纵轴垂直']['score']
        flag4_0 = level_10(row[4])
        flag4_1 = level_10(score)
        manual_line_is_horizontal.append(flag4_0)
        model_line_is_horizontal.append(flag4_1)

        score = details['下缘包含尺桡骨3-5cm']['score']
        flag5_0 = level_5(row[5])
        flag5_1 = level_5(score)
        manual_include_radius_ulna.append(flag5_0)
        model_include_radius_ulna.append(flag5_1)

        score = details['左右最外侧距影像边缘3-5cm']['score']
        flag6_0 = level_6(row[6])
        flag6_1 = level_6(score)
        manual_distance_to_edge.append(flag6_0)
        model_distance_to_edge.append(flag6_1)

        flag3s.append(abs(details['尺桡骨茎突连线中点位于图像正中']['score'] - row[3]))
        flag4s.append(abs(details['尺桡骨茎突连线与影像纵轴垂直']['score'] - row[4]))
        flag5s.append(abs(details['下缘包含尺桡骨3-5cm']['score'] - row[5]))
        flag6s.append(abs(details['左右最外侧距影像边缘3-5cm']['score'] - row[6]))

    print('%.2f ± %.2f' % (np.mean(flag3s), np.std(flag3s)))
    print('%.2f ± %.2f' % (np.mean(flag4s), np.std(flag4s)))
    print('%.2f ± %.2f' % (np.mean(flag5s), np.std(flag5s)))
    print('%.2f ± %.2f' % (np.mean(flag6s), np.std(flag6s)))

    # Used to filter various types of images in the confusion matrix
    for idx, (i, j) in enumerate(zip(model_scores, manual_scores)):
        if i == 0 and j == 1:
            print(rows[idx][0])
            shutil.copy(os.path.join('/data/experiments/wrist-landmark/AP/visualized_results', rows[idx][0] + '.png'),
                        os.path.join('/data/temp', rows[idx][0] + '.png'))
    cnt = 0
    for i, j in zip(manual_scores, model_scores):
        if i == j:
            cnt += 1
    print('total:', cnt / len(manual_scores))

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
    color = 'YlOrBr'

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
    plt.savefig('./compare/AP-scores.png')

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
    plt.savefig('./compare/AP-artifact.png')

    # draw ~ of joint
    plt.clf()
    CM_joint = metrics.confusion_matrix(model_joint, manual_joint)
    hm = sns.heatmap(CM_joint, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Metacarpophalangeal Joint', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/AP-joint.png')

    # draw ~ of midpoint
    plt.clf()
    CM_midpoint = metrics.confusion_matrix(model_midpoint_is_center, manual_midpoint_is_center, labels=[0, 1, 2])
    labels = ['Fail', 'Poor', 'Good']
    hm = sns.heatmap(CM_midpoint, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Midpoint of Styloid Process', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/AP-midpoint.png')

    # draw ~ of line
    plt.clf()
    CM_line = metrics.confusion_matrix(model_line_is_horizontal, manual_line_is_horizontal, labels=[0, 1, 2, 3])
    labels = ['Fail', 'Poor', 'Average', 'Excellent']
    hm = sns.heatmap(CM_line, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Line of Styloid Process', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/AP-line.png')

    # draw ~ of include
    plt.clf()
    CM_include = metrics.confusion_matrix(model_include_radius_ulna, manual_include_radius_ulna, labels=[0, 1, 2, 3])
    hm = sns.heatmap(CM_include, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of \nthe Raidus and Ulna', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/AP-include.png')

    # draw ~ of distance
    plt.clf()
    CM_distance = metrics.confusion_matrix(model_distance_to_edge, manual_distance_to_edge, labels=[0, 1, 2, 3])
    hm = sns.heatmap(CM_distance, annot=True, fmt='d', annot_kws=annot_kws, xticklabels=labels, yticklabels=labels,
                     cmap=color)
    hm.set_xticklabels(hm.get_xticklabels(), fontsize=labelsize)
    hm.set_yticklabels(hm.get_yticklabels(), fontsize=labelsize)
    hm.invert_yaxis()
    plt.xlabel('Manual', font1)
    plt.ylabel('Model', font1)
    plt.title('Confusion Matrix of\nthe Distance', fontsize=title_font_size, loc='center')
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('./compare/AP-distance.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='compare manual eval with model eval')
    parser.add_argument('--model-eval', type=str, default='../inference_result/inference_AP.json')
    parser.add_argument('--manual-eval', type=str, default='./manual_evaluation_AP.xlsx')
    args = parser.parse_args()
    main(args.model_eval, args.manual_eval)
