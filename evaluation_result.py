import json
import os
import sys

from Metric.THVC import *
import zero_shot_video_caption_Hallucination
from Metric.THVC.Thinking_Hallucination_for_Video_Captioning import evaluation_coaha
from Metric.coco_caption.my_eval.eval_script import coco_evaluation


def get_msvd_label(source_filepath, write_filepath):
    """
    :param source_filepath: 输入的原注释文件
    :param write_filepath: 格式化后的输出文件
    :return: 返回格式化的标签数据
    格式为[{'video_path': video_name, "captions": caption_list}]
    """

    file_path = source_filepath

    with open(file_path, "r") as file:
        lines = file.readlines()
        file.close()
    # 去除每行的换行符和空白字符
    lines = [line.strip() for line in lines]

    label_list = []
    video_name = ""
    caption_list = []
    for line in lines:
        item = line.split(" ", 1)
        if item[0] == video_name:
            caption_list.append(item[1])
        if video_name == "":
            video_name = item[0]
            caption_list.append(item[1])
        if item[0] != video_name and video_name != "":
            label_list.append({'video_path': video_name, "captions": caption_list})
            caption_list = []
            video_name = item[0]
            caption_list.append(item[1])
    label_list.append(caption_list)
    print(len(label_list))

    with open(write_filepath, "w", encoding="utf-8") as write_file:
        json.dump(label_list, write_file)
        write_file.close()
    return label_list


def evaluation_CoCo_caption(pre, label):

    result, score = coco_evaluation("result/zero_shot_video_caption_all_pre.txt", "result/msvd_video_caption1.txt", 0)

    return result, score


def evaluation_Clip_S():
    return []


def evaluation_BLip_S():

    return []


def evaluation_PAC():

    return []


def evaluation_Cocha():
    pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_mapping("result/youtube_mapping.txt",
                                                                                 "result/zero_shot_video_caption_all_pre.txt",
                                                                                 "1301", "")
    cocha = evaluation_coaha(pre_dict, "msvd")
    print(cocha)
    return cocha





if __name__ == '__main__':
    """
    # 读取获得的
    msvd_caption_pre_result = []
    with open("result/zero_shot_video_caption_all_pre.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        json_data = []
        for caption in data:
            print(caption)
    """
    # 读取
    # get_msvd_label("/home/no-zjc/datasets/MSVD-QA/msvd_video_caption.txt", "result/msvd_label_file.json")

    # evaluation_CoCo_caption("", "")
    evaluation_Cocha()


