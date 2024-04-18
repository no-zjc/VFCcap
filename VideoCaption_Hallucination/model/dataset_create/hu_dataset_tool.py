# coding=utf-8
"""
    @project: few-shot-video-to-text
    @Author：no-zjc
    @file： hu_dataset_tool.py
    @date：2024/1/31 3:37
"""
import json
import random

import h5py
from tqdm import tqdm
import torch
import clip
import numpy as np

from VideoCaption_Hallucination.File_Tools import File_Tools
from nltk.stem import WordNetLemmatizer

# 0代表MSVD/1代表MSRVTT
dt = 1
dataset_type = ['msvd', 'msrvtt']
dataset_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/", "/home/wy3/zjc_data/datasets/MSR-VTT/"]
data_fts_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_clip_ViT_L_14_fts_four.hdf5",
                 "/home/wy3/zjc_data/datasets/MSR-VTT/msr-vtt_clip_ViT_L_14_fts_four.hdf5"]
mapping_file = ["/home/wy3/zjc_data/datasets/MSVD-QA/youtube_mapping.txt"]
annotations_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_caption_format.json",
                    "/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json"]
captions_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/msvd_combine_frame_retrieved_caps_Vit_B32.json",
                 "/home/wy3/zjc_data/datasets/MSR-VTT/msrvtt_combine_frame_retrieved_caps_Vit_B32.json"]
data_split = ["train", "val", "test"]

dataset_type = dataset_type[dt]
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

classification_type = 3
label_smoothing_epsilon = 0.2
tem_p = 2


def load_data_for_training_HU(annot_path):
    annotations = json.load(open(annot_path))
    data = {'train': [], 'val': [], 'test': []}
    for item in annotations:
        samples = []
        samples.append(
                {'id': item["id"], 'vid': item["vid"], 'fts_key': item["fts_key"], "text_hallucination": item["text_hallucination"], "text_label": item["text_label"], "label_list": item["label_list"], "hallucination_type_text": item["hallucination_type_text"]})
        if item["split"] == "train":
            data['train'] += samples
        elif item["split"] == "val":
            data['val'] += samples
        elif item["split"] == "test":
            data['test'] += samples

    return data


def compute_cosine_similarity(video_fts, cap_fts):
    similarities = []
    for img_fts in video_fts:
        # 计算点积
        dot_product = torch.sum(img_fts * cap_fts)

        # 计算向量的范数
        img_norm = torch.norm(img_fts)
        cap_norm = torch.norm(cap_fts)

        # 计算余弦相似度
        cosine_similarity = dot_product / (img_norm * cap_norm)
        similarities.append(cosine_similarity)

    similarities_tensor = torch.tensor(similarities)
    similarity = torch.mean(similarities_tensor)
    similarity_float = similarity.item()
    return similarity_float

def random_choice():
    choices = [0, 1, 2, 3]
    probabilities = [0.2, 0.1, 0.35, 0.35]

    choice = random.choices(choices, probabilities)[0]
    return choice

def is_same_word(word1, word2):
    lemmatizer = WordNetLemmatizer()
    base_word1 = lemmatizer.lemmatize(word1)
    base_word2 = lemmatizer.lemmatize(word2)
    return base_word1 == base_word2
def load_data_for_training_MSVD(mapping_path, annot_path, caps_path=None):
    annotations = json.load(open(annot_path))
    tid = 1
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))

    data = {'train': [], 'val': []}
    with open(mapping_path, 'r') as file:
        for line in file:
            words = line.split()

            file_name = words[0] + ".avi"
            vid = words[1].split("\n")[0]
            vid_int = int(vid.split("vid")[1])

            if caps_path is not None:
                caps = retrieved_caps[str(vid_int)]
            else:
                caps = None

            samples = []
            for item in annotations.get(words[0]):
                samples.append(
                    {'id': tid, 'file_name': file_name, 'vid': words[1], 'vid_int': vid_int, 'caps': caps, 'text': item, 'fts_key': words[0]})
                tid = tid + 1
            if vid_int <= 1200:
                # if vid_int <= 10:
                data['train'] += samples
            else:
                data['val'] += samples
    return data


def load_data_for_training_MSR_VTT(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))
    tid = 1
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': [], 'test': []}
    for i, (key, value) in enumerate(annotations.items()):
        file_name = key + ".mp4"
        vid_int = int(key.split("video")[1])

        if caps_path is not None:
            caps = retrieved_caps[str(vid_int)]
        else:
            caps = None

        samples = []
        for item in value:
            samples.append(
                {'id': tid, 'file_name': file_name, 'vid': key, 'vid_int': vid_int, 'caps': caps, 'text': item, 'fts_key': key})
            tid = tid + 1
        if vid_int < 6513:
            data['train'] += samples
        elif vid_int < 7010:
            data['val'] += samples
        elif vid_int < 10000:
            data['test'] += samples

    return data

def get_smoothing_label(label_list, fts_key, fts_data, hu_sentences, clip_model):
    video_fts = fts_data["video_fts"][fts_key][()]
    video_fts = torch.tensor(video_fts, dtype=torch.float, device=device)
    with torch.no_grad():
        input_ids = clip.tokenize(hu_sentences)
        hu_caption_clip = clip_model.encode_text(input_ids)

    video_caption_cos = compute_cosine_similarity(video_fts, hu_caption_clip)

    for i in range(len(label_list)):
        if i == 0:
            label_list[i] = (1 - label_smoothing_epsilon) * label_list[i] + label_smoothing_epsilon * video_caption_cos * tem_p
        else:
            label_list[i] = (1 - label_smoothing_epsilon) * label_list[i] + (label_smoothing_epsilon - (label_smoothing_epsilon * video_caption_cos * tem_p)) / (classification_type - 1)

    return label_list, video_caption_cos

def get_features_labels_(split_text, clip_model, a_path, fts_path):
    data = load_data_for_training_HU(a_path)
    hu_data = data[split_text]
    h5_features = h5py.File(fts_path, 'r')
    hu_features = []
    hu_labels = []
    for item in tqdm(hu_data):
        hu_labels.append(item["label_list"])
        with torch.no_grad():
            input_ids = clip.tokenize(item["text_hallucination"]).to(device)
            hu_caption_clip = clip_model.encode_text(input_ids).cpu().numpy()
        video_fts = h5_features["video_fts"][item["fts_key"]][()]
        padded_arr = np.pad(video_fts, [(0, 80 - video_fts.shape[0]), (0, 0)], mode='constant')
        hu_f = np.vstack((hu_caption_clip, padded_arr))
        hu_features.append(hu_f)
    return hu_data, np.array(hu_features), np.array(hu_labels)


def dataset_create():
    # video_uts = VideoUtils()
    if dataset_type == 'msvd':
        data = load_data_for_training_MSVD(mapping_file[0], annotations_path[0])
    elif dataset_type == 'msrvtt':
        data = load_data_for_training_MSR_VTT(annotations_path[1])

    clip_model, feature_extractor = clip.load("ViT-L/14", device=device, download_root='../../../clip_checkpoints')
    video_fts_data = h5py.File(data_fts_path[dt], 'r')

    min_cos = 0
    max_cos = 0

    train_data = data['train']
    object_dict = File_Tools.load_json_data(dataset_path[dt] + "action_object/" + dataset_type + "_object_dict.json")
    action_dict = File_Tools.load_json_data(dataset_path[dt] + "action_object/" + dataset_type + "_action_dict.json")
    all_object_list = File_Tools.load_json_data(dataset_path[dt] + "action_object/" + dataset_type + "_object_list.json")
    all_action_list = File_Tools.load_json_data(dataset_path[dt] + "action_object/" + dataset_type + "_action_list.json")

    # video的数据集划分，{{video_fts_key:split_text}}
    video_split_dict = {}

    hu_dataset = []
    for anno_info in tqdm(train_data):
        object_list = [x for x in object_dict[anno_info["vid"]] if len(x) > 1]
        action_list = [x for x in action_dict[anno_info["vid"]] if len(x) > 1]
        other_object_list = [x for x in all_object_list if x not in object_list and len(x) > 1]
        other_action_list = [x for x in all_action_list if x not in action_list and len(x) > 1]
        text_list = anno_info["text"].split(" ")
        object_index = []
        action_index = []
        for i in range(len(text_list)):
            label_flag = False
            for ol in object_list:
                if (ol == text_list[i] or is_same_word(ol, text_list[i])) and len(ol) > 1:
                    object_index.append(i)
                    label_flag = True
                    break
            if label_flag:
                continue
            for al in action_list:
                if (al == text_list[i] or is_same_word(al, text_list[i]) or text_list[i].startswith(al)) and len(al) > 1:
                    action_index.append(i)
                    label_flag = True
                    break
            if label_flag:
                continue

        random_type = random_choice()
        hu_type = ["no_object_hallucination", "no_action_hallucination", "object_hallucination", "action_hallucination"]
        text_hu = ""
        if video_split_dict.get(anno_info["fts_key"]) is not None:
            split_text = video_split_dict.get(anno_info["fts_key"])
        else:
            split_text = random.choices(data_split, [0.8, 0.1, 0.1])[0]
            video_split_dict.update({anno_info["fts_key"]: split_text})
        text_label = ""
        label_list = []
        hu_type_text = ""
        # no_action_hallucination
        if random_type == 1:
            if len(action_index) < 1:
                random_type = 0
            else:
                hu_position = random.choices(action_index)[0]
                text_hu = " ".join(text_list[:hu_position+1])
                text_label = text_hu
                label_list = [1, 0, 0]
                hu_type_text = hu_type[1]

        # no_object_hallucination
        if random_type == 0:
            if len(object_index) < 1:
                random_type = 3
            else:
                hu_position = random.choices(object_index)[0]
                text_hu = " ".join(text_list[:hu_position+1])
                text_label = text_hu
                label_list = [1, 0, 0]
                hu_type_text = hu_type[0]

        # action_hallucination
        if random_type == 3:
            if len(action_index) < 1:
                random_type = 2
            else:
                hu_position = random.choices(action_index)[0]
                hu_word = random.choices(other_action_list)[0]
                text_hu = " ".join(text_list[:hu_position]) + " " + hu_word
                text_label = " ".join(text_list[:hu_position + 1])
                label_list = [0, 0, 1]
                hu_type_text = hu_type[3]

        # object_hallucination
        if random_type == 2:
            if len(object_index) < 1:
                continue
            else:
                hu_position = random.choices(object_index)[0]
                hu_word = random.choices(other_object_list)[0]
                text_hu = " ".join(text_list[:hu_position]) + " " + hu_word
                text_label = " ".join(text_list[:hu_position + 1])
                label_list = [0, 1, 0]
                hu_type_text = hu_type[2]

        anno_info.update({"text_hallucination": text_hu})
        anno_info.update({"text_label": text_label})
        label_list, COS_VT = get_smoothing_label(label_list, anno_info["fts_key"], video_fts_data, text_hu, clip_model)
        if COS_VT < min_cos:
            min_cos = COS_VT
        if COS_VT > max_cos:
            max_cos = COS_VT
        anno_info.update({"label_list": label_list})
        anno_info.update({"hallucination_type_text": hu_type_text})
        anno_info.update({"split": split_text})
        hu_dataset.append(anno_info)
        continue

    File_Tools.write_to_json(dataset_path[dt] + dataset_type + "_hallucination_dataset_sv_ls.json", hu_dataset)
    print("数据文件已经保存到" + dataset_path[dt] + dataset_type + "_hallucination_dataset_sv_ls.json")

    print("min:" + str(min_cos))
    print("max:" + str(max_cos))



if __name__ == '__main__':

    dataset_create()


