import argparse
import logging
import threading

import clip
from model.CapGenerator import CLIPTextGenerator
import torch
import os
# from data_loader import VideosDataset, ImagesDataset, ImagesPairsDataset
from datetime import datetime
import shutil
import json
import sys
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from VideoCaption_Hallucination.File_Tools import File_Tools as FT



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomized_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--db_filter_path", type=str, default=None, help="file to filter db items, e.g karpathy split")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=20)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--token_wise", action="store_true", help="Should we step the optimization at each token gen")
    parser.add_argument("--num_dummy_tokens", type=int, default=5)
    parser.add_argument("--sentence_iterations", type=int, default=30)
    parser.add_argument("--sampling_top_k", type=int, default=3)
    parser.add_argument("--db_start_idx", type=int, default=0)
    parser.add_argument("--db_num_images", type=int, default=0)
    parser.add_argument("--clip_loss_temperature", type=float, default=1.0)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.006)
    parser.add_argument("--scheduler_type", type=CLIPTextGenerator.SchedType, default='cosine')
    parser.add_argument("--weight_decay_scale", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help='How much much to deter deter repeats')
    parser.add_argument("--entity_penalty", type=float, default=2, help='How much to deter CapsLock in middle of sent')
    parser.add_argument("--ending_bonus", type=float, default=2, help='How much to help the sentence to end')
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--pairs_path", type=str, default="")
    parser.add_argument("--batch_num", type=int, default="1")

    # parser.add_argument('--data_path', type=str, default='examples/example_video.mp4')
    parser.add_argument('--data_path', type=str, default='/home/no-zjc/datasets/MSVD-QA/video/-Cv5LsqKUXc_17_25.avi')
    parser.add_argument('--run_type',
                        default='caption_videos',
                        nargs='?',
                        choices=['caption_images', 'caption_videos'])
    return parser


def filter_video(image_fts, similiarities):
    THRESHOLD = 0.9
    groups = []
    curr_group = []
    for i in range(similiarities.size(0)):
        if len(curr_group) == 0:
            curr_group.append(i)

        if i + 1 == similiarities.size(0):
            if len(curr_group) >= 1:
                groups.append(curr_group)
            break

        if similiarities[curr_group[0]][i + 1] > THRESHOLD:
            curr_group.append(i + 1)
        else:
            if len(curr_group) >= 1:
                groups.append(curr_group)
            curr_group = []

    result_features = []
    selected_indices = []
    if len(groups) >= 1:
        for i, group in enumerate(groups):
            result_features.append(image_fts[group[0]])
            selected_indices.append(group[0])

    return torch.stack(result_features), selected_indices


def get_clip_video_frames(video_path, clip_preprocess):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    sample_time = FPS // 3
    imgs = []

    i = 0
    while (cap.isOpened()):
        ret, cv2_im = cap.read()

        if ret and i % sample_time == 0:
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(converted)
            imgs.append(pil_im)
        elif not ret:
            break

        i += 1

    cap.release()

    images = torch.cat([clip_preprocess(x).unsqueeze(0) for x in imgs])

    return images


def get_clip_image(image_path, clip_preprocess):
    images = torch.cat([clip_preprocess(Image.open(image_path)).unsqueeze(0)])

    return images


def run_video(args, video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_generator = CLIPTextGenerator(**vars(args))
    # 编码视频帧
    video_frames = get_clip_video_frames(video_path, text_generator.clip_preprocess).to(device)

    with torch.no_grad():
        frames_fts = text_generator.clip.encode_image(video_frames).detach()
        print(frames_fts.shape)
        frames_fts = torch.nn.functional.normalize(frames_fts, dim=-1).detach()
        print(frames_fts.shape)

        similiarities = frames_fts @ frames_fts.T
        image_fts, selected_frames_indices = filter_video(frames_fts, similiarities)
        print(image_fts.shape, selected_frames_indices)

    clip_sorted_captions, mixed_sorted_captions, decoded_options, beam_caps = text_generator.generate(image_fts)

    print(clip_sorted_captions)

    return clip_sorted_captions[0]


def run_image(args, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_generator = CLIPTextGenerator(**vars(args))

    image = get_clip_image(image_path, text_generator.clip_preprocess).to(device)

    with torch.no_grad():
        image_fts = text_generator.clip.encode_image(image).detach()
        image_fts = torch.nn.functional.normalize(image_fts, dim=-1).detach()

    clip_sorted_captions, mixed_sorted_captions, decoded_options, beam_caps = text_generator.generate(image_fts)

    print(clip_sorted_captions)

    return clip_sorted_captions[0]


def get_file_names(directory):
    file_names = []

    # 遍历目录中的所有文件和文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(os.path.join(root, file))

    return file_names


def get_caption_json(start, wfilename, cli_args):
    # 指定目录路径
    directory = "/home/no-zjc/datasets/MSVD-QA/video/"

    # 获取指定目录下的所有文件名
    file_names = get_file_names(directory)
    caption_result = []
    flag = 0
    restart = 0
    # 输出文件名
    for file_name in file_names:
        if file_name == file_names[start]:
            restart = 1
        if restart == 0:
            continue
        flag = flag + 1
        # torch.set_num_threads(3)

        if cli_args.run_type == 'caption_videos':
            caption_b = run_video(cli_args, file_name)
            caption = {"video_path": file_name, "caption": caption_b}
            print(caption)
            caption_result.append(caption)

        if ((flag % 10 == 0) and (flag != 0)) or file_name == file_names[start + 401]:
            # 将数据写入JSON文件
            with open(wfilename, "w", encoding="utf-8") as json_file:
                json.dump(caption_result, json_file)


def montage_caption_file(filelist, result_filename):
    last_caption = {}
    caption_result = []
    montage_flag = 1

    for filename in filelist:
        with open(filename, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            for caption in data:
                if montage_flag == 1:
                    caption_result.append(caption)
                if caption == last_caption:
                    montage_flag = 1

            last_caption = data[-1]
            montage_flag = 0
            json_file.close()

    with open(result_filename, "w", encoding="utf-8") as write_file:
        json.dump(caption_result, write_file)
        write_file.close()
    return caption_result


def json2txt(result_file):
    with open(result_file, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        json_file.close()
    with open(result_file[:result_file.index('.')] + ".txt", "w", encoding="utf-8") as txtfile:
        for caption in data:
            filename = caption["video_path"].split("/")[-1]
            caption = caption["caption"]
            txtfile.write(filename.split(".", 1)[0] + "\t" + caption + '\n')
    txtfile.close()

def txt2txt(result_file):
    file_path = result_file

    with open(file_path, "r") as file:
        lines = file.readlines()
        file.close()
    # 去除每行的换行符和空白字符
    lines = [line.strip() for line in lines]

    with open(result_file[:result_file.index('.')] + "1.txt", "w") as file1:
        for line in lines:
            item = line.split(" ", 1)
            file1.write(item[0] + "\t" + item[1] + '\n')


def msvd_caption_result_mapping(mapping_file, result_file, start, savefile=""):
    mapping_d = []
    pre_d = []
    pre_dict = {}
    flag = 0
    with open(mapping_file, "r") as mf:
        lines = mf.readlines()
        mf.close()
    for line in lines:
        line = line.strip()
        mapping_d.append(line.split(" "))


    with open(result_file, "r", encoding="utf-8") as t_file:
        lines_r = t_file.readlines()
        t_file.close()
    for line_r in lines_r:
        line_r = line_r.strip()
        pre_d.append(line_r.split("\t"))


    for mapping in mapping_d:
        if mapping[1].split("d")[1] == start:
            flag = 1
        if flag != 1:
            continue
        for pre in pre_d:
            if mapping[0] == pre[0]:
                pre_dict.update({mapping[1]: [pre[1]]})


    # with open(savefile, "w", encoding="utf-8") as s_file:
    #     json.dump(pre_dict, s_file)
    #     s_file.close()
    # print(pre_dict)

    return pre_dict


def msvd_caption_result_format(result_file, start=1201):
    pre_dict = {}
    pre_list = FT.load_json_data(result_file)
    for pre in pre_list:
        key = "vid" + str(pre.get("video_id"))
        value = [pre.get("caption")]
        pre_dict.update({key: value})

    return pre_dict


def msrvtt_caption_result_format(result_file, start=1201):
    pre_dict = {}
    pre_list = FT.load_json_data(result_file)
    for pre in pre_list:
        key = pre.get("file_name")
        value = [pre.get("caption")]
        pre_dict.update({key: value})

    return pre_dict

if __name__ == "__main__":
    """
    # 指定目录路径
    directory = "/home/no-zjc/datasets/MSVD-QA/video/"

    # 获取指定目录下的所有文件名
    file_names = get_file_names(directory)
    caption_result = []
    flag = 0
    restart = 0
    # 输出文件名
    for file_name in file_names:
        if file_name == "/home/no-zjc/datasets/MSVD-QA/video/hJFBXHtxKIc_286_291.avi":
            restart = 1
            print(str(flag) + "/" + str(len(file_names)))
        if restart == 0:
            continue
        flag = flag + 1
        # torch.set_num_threads(3)
        cli_args = get_parser().parse_args()


        if cli_args.run_type == 'caption_videos':
            caption_b = run_video(cli_args, file_name)
            caption = {"video_path": file_name, "caption": caption_b}
            print(caption)
            caption_result.append(caption)


        if((flag % 10 == 0) and (flag != 0)) or file_name == file_names[-1]:
            # 将数据写入JSON文件
            with open("result/zero_shot_video_caption_pre1.json", "w", encoding="utf-8") as json_file:
                json.dump(caption_result, json_file)


    # 数据将分为多个任务进行计算
    cli_args = get_parser().parse_args()
    if cli_args.batch_num == 1:
        get_caption_json(368, "11.json", cli_args)

    if cli_args.batch_num == 2:
        get_caption_json(768, "12.json", cli_args)

    if cli_args.batch_num == 3:
        get_caption_json(1168, "13.json", cli_args)

    if cli_args.batch_num == 4:
        get_caption_json(1568, "14.json", cli_args)

    """
    """
    # 拼接所有的输出caption文件并输入到json文件中
    filelist = ["result/zero_shot_video_caption_pre.json", "result/11.json", "result/12.json", "result/13.json", "result/14.json"]
    result_filename = "result/zero_shot_video_caption_all_pre.json"
    result = montage_caption_file(filelist, result_filename)
    print(len(result))
    """

    # 格式化json为TXT
    # json2txt("result/zero_shot_video_caption_all_pre.json")

    # txt2txt("result/msvd_video_caption.txt")

    msvd_caption_result_mapping("result/youtube_mapping.txt", "result/zero_shot_video_caption_all_pre.txt", "1301", " ")
