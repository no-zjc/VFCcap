# coding=utf-8
"""
    @project: zero-shot-video-to-text-main
    @Author：no-zjc
    @file： video_utils.py
    @date：2023/10/30 16:35
"""
import os

import h5py
import torch
import cv2
from PIL import Image
import clip
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Dataset

from VideoCaption_Hallucination.File_Tools import File_Tools

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This video shows "


def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True
        truncation = True

    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id]
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))

    if is_test:
        return input_ids
    else:
        return input_ids, label_ids



def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred


class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.simple_prefix = "This video shows "

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                      + max_caption_length * k  # retrieved captions
                                      + len(tokenizer.encode(self.template))  # template
                                      + len(tokenizer.encode('\n\n')) * (k - 1)  # separator between captions
                                      )
            assert k is not None
            self.k = k
        else:
            self.max_target_length = max_caption_length + len(tokenizer.encode(self.simple_prefix))
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag:
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)

        # combine_fts = VideoUtils.get_combine_features(self.features, self.df['fts_key'][idx])
        # encoding = {"encoder_outputs": combine_fts,
        #             "decoder_input_ids": torch.tensor(decoder_input_ids),
        #             "labels": torch.tensor(labels)}

        output_tensor, action_fts_o, video_obj_fts_m = VideoUtils.get_all_features(self.features, self.df['fts_key'][idx])

        encoding = {"encoder_outputs": output_tensor,
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels),
                    "action_fts_o": action_fts_o,
                    "video_obj_fts_m": video_obj_fts_m,
                    }

        return encoding


class VideoUtils:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device,
        #                                             download_root='../clip_checkpoints', jit=False)

        self.clip, self.clip_preprocess = clip.load("ViT-L/14", device=self.device,
                                                    download_root='../clip_checkpoints', jit=False)
        self.clip = self.clip.eval()
        # Freeze CLIP weights
        for param in self.clip.parameters():
            param.requires_grad = False

    @staticmethod
    def get_triplet_features(features, fts_key, video_path=None):
        # load precomputed features
        # 最大的特征维度，特征默认max_fts_dim*512
        max_fts_dim = 80
        # 动作特征池化的最大前后帧范围[kfs-max_action_dis,kfs+max_action_dis]
        max_action_dis = 5

        # 对象特征最大值
        max_obj = 5

        # 定义填充量为负无穷，填充到 max_fts_dimx512
        # padding = torch.tensor(float('-inf'))
        padding = torch.tensor(0)

        action_fts = []
        object_fts = []
        if video_path is not None:
            video_fts, video_kfs, video_full_fts = videoUtils.get_clip_video_feats(video_path)
        else:
            video_full_fts = features["video_full_fts"][fts_key][()]
            video_fts = features["video_fts"][fts_key][()]
            video_kfs = features["video_kfs"][fts_key][()]
            video_obj_fts_m = features["video_obj_fts"][fts_key][()][:max_obj]
            video_obj_fts = np.reshape(video_obj_fts_m, (max_obj * 4, 512))
        for i in range(len(video_kfs)):
            st = 0
            ed = len(video_full_fts) - 1
            if i - 1 >= 0:
                st = video_kfs[i - 1] + 1
            if i + 1 < len(video_kfs):
                ed = video_kfs[i + 1] - 1

            if video_kfs[i] - st > max_action_dis:
                st = video_kfs[i] - max_action_dis
            if ed - video_kfs[i] > max_action_dis:
                st = video_kfs[i] + max_action_dis

            pooled_features = np.mean(video_full_fts[st:ed + 1], axis=0)
            action_fts.append(pooled_features)
        action_fts = np.stack(action_fts, axis=0)
        action_fts = torch.tensor(action_fts)
        action_fts = action_fts.to(torch.float32)
        action_fts_o = torch.nn.functional.pad(action_fts, (0, 0, 0, max_fts_dim - video_fts.shape[0]), value=padding)

        object_fts = np.stack(video_obj_fts, axis=0)
        object_fts = torch.tensor(object_fts)
        object_fts_o = object_fts.to(torch.float32)

        video_obj_fts_m = np.stack(video_obj_fts_m, axis=0)
        video_obj_fts_m = torch.tensor(video_obj_fts_m)
        video_obj_fts_m = video_obj_fts_m.to(torch.float32)

        input_tensor = torch.tensor(video_fts)
        input_tensor = input_tensor.to(torch.float32)
        output_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, max_fts_dim - video_fts.shape[0]),
                                                value=padding)
        return output_tensor, object_fts_o, action_fts_o, video_obj_fts_m

    @staticmethod
    def get_combine_features(features, fts_key, video_path=None):

        output_tensor, object_fts_o, action_fts_o, video_obj_fts_m = VideoUtils.get_triplet_features(features, fts_key, video_path=None)

        # combine_fts = torch.cat([output_tensor, object_fts_o, action_fts_o], dim=0)
        combine_fts = torch.cat([output_tensor], dim=0)
        return combine_fts

    @staticmethod
    def get_all_features(features, fts_key, video_path=None):

        output_tensor, object_fts_o, action_fts_o, video_obj_fts_m = VideoUtils.get_triplet_features(features, fts_key, video_path=None)
        output_tensor, action_fts_o, video_obj_fts_m = torch.cat([output_tensor], dim=0), torch.cat([action_fts_o], dim=0), torch.cat([video_obj_fts_m], dim=0)
        return output_tensor, action_fts_o, video_obj_fts_m

    def filter_video(self, image_fts, similiarities):
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

    def get_clip_video_frames(self, video_path, clip_preprocess):
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

    def get_clip_video_feats(self, video_path):
        # 编码视频帧
        video_frames = self.get_clip_video_frames(video_path, self.clip_preprocess).to(self.device)

        with torch.no_grad():
            frames_fts = self.clip.encode_image(video_frames).detach()
            # print(frames_fts.shape)
            frames_fts = torch.nn.functional.normalize(frames_fts, dim=-1).detach()
            # print(frames_fts.shape)

            similiarities = frames_fts @ frames_fts.T
            image_fts, selected_frames_indices = self.filter_video(frames_fts, similiarities)
            # print(image_fts.shape, selected_frames_indices)
        return image_fts, selected_frames_indices, frames_fts

    def create_feature_hdf5(self, video_path, h5file_save_path):
        """
        clip提取视频帧特征并将其保存到hdf5中，
        包含三个组（video_full_fts【全部每一秒三帧视频序列clip特征】, video_fts【经过阈值筛选后的视频特征】, video_kfs【视频帧序号】）,key为视频的基础文件名
        :param video_path: "/home/no-zjc/datasets/MSVD-QA/video/"
        :param h5file_save_path: "/home/no-zjc/datasets/MSVD-QA/" + 'msvd_video_random_length_fts.hdf5'
        :return:
        """
        print("特征向量提取中")
        with h5py.File(h5file_save_path, 'w') as h5py_file:
            video_full_fts = h5py_file.create_group("video_full_fts")
            video_fts = h5py_file.create_group("video_fts")
            video_kfs = h5py_file.create_group("video_kfs")
            full_paths, file_names, base_names = File_Tools.get_filenames(video_path)
            for i in tqdm(range(len(full_paths))):
                image_fts, selected_frames_indices, full_images_fts = videoUtils.get_clip_video_feats(
                    video_path + file_names[i])
                video_fts.create_dataset(str(base_names[i]), data=image_fts.cpu().numpy())
                video_kfs.create_dataset(str(base_names[i]), data=selected_frames_indices)
                video_full_fts.create_dataset(str(base_names[i]), data=full_images_fts.cpu().numpy())
            h5py_file.close()
        print("特征文件已生成")

    def get_max_fts_dim(self, video_path, h5file_save_path):
        full_paths, file_names, base_names = File_Tools.get_filenames(video_path)
        base_names.sort()
        with h5py.File(h5file_save_path, 'r') as h5py_file_r:
            video_fts = h5py_file_r["video_fts"]
            video_kfs = h5py_file_r["video_kfs"]
            video_full_fts = h5py_file_r["video_full_fts"]

            max = 0
            for base_name in base_names:
                da = np.array(video_full_fts[base_name])
                if da.shape[0] > max:
                    max = da.shape[0]
            print(max)

    def load_data_for_training_MSVD(self, mapping_path, annot_path, caps_path=None):
        annotations = json.load(open(annot_path))
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
                        {'file_name': file_name, 'vid': vid_int, 'caps': caps, 'text': item, 'fts_key': words[0]})

                if vid_int <= 1200:
                # if vid_int <= 10:
                    data['train'] += samples
                else:
                    data['val'] += samples
        return data

    def load_data_for_inference_MSVD(self, mapping_path, annot_path, caps_path=None):
        annotations = json.load(open(annot_path))
        if caps_path is not None:
            retrieved_caps = json.load(open(caps_path))

        data = {'train': [], 'val': [], 'test': []}
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

                video = []

                video.append(
                    {'file_name': file_name, 'vid': vid_int, 'text': annotations.get(words[0])[0], 'caps': caps,
                     'fts_key': words[0]})

                if vid_int <= 1200:
                    data['train'] += video
                elif vid_int <= 1300:
                    data['val'] += video
                else:
                    data['test'] += video
        return data

    def load_data_for_training_MSR_VTT(self, annot_path, caps_path=None):
        annotations = json.load(open(annot_path))
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
                    {'file_name': file_name, 'vid': vid_int, 'caps': caps, 'text': item, 'fts_key': key})

            if vid_int < 6513:
                data['train'] += samples
            elif vid_int < 7010:
                data['val'] += samples
            elif vid_int < 10000:
                data['test'] += samples

        return data

    def load_data_for_inference_MSR_VTT(self, annot_path, caps_path=None):
        annotations = json.load(open(annot_path))
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
            samples.append(
                {'file_name': file_name, 'vid': vid_int, 'caps': caps, 'text': value[0], 'fts_key': key})

            if vid_int < 6513:
                data['train'] += samples
            elif vid_int < 7010:
                data['val'] += samples
            elif vid_int < 10000:
                data['test'] += samples
        return data

    def add_obj_feature_hdf5(self, video_path, old_path, h5file_save_path):
        """
        clip提取视频帧特征并将其保存到hdf5中，
        包含三个组（video_full_fts【全部每一秒三帧视频序列clip特征】, video_fts【经过阈值筛选后的视频特征】, video_kfs【视频帧序号】）,key为视频的基础文件名
        :param video_path: "/home/no-zjc/datasets/MSVD-QA/video/"
        :param h5file_save_path: "/home/no-zjc/datasets/MSVD-QA/" + 'msvd_video_random_length_fts.hdf5'
        :return:
        """
        msvd_p = "/home/wy3/zjc_data/datasets/MSVD-QA/"
        msrvtt_p = "/home/wy3/zjc_data/datasets/MSR-VTT/"

        msvd_obj = ["MSVD_vg_objects_train.hdf5", "MSVD_vg_objects_valid.hdf5", "MSVD_vg_objects_test.hdf5"]
        msrvtt_obj = ["MSRVTT_vg_objects_train.hdf5", "MSRVTT_vg_objects_valid.hdf5", "MSRVTT_vg_objects_test.hdf5"]

        old_data = h5py.File(old_path, 'r')
        new_data_list = []
        if "msvd" in old_path:
            new_data_list.append(h5py.File(msvd_p + msvd_obj[0], 'r'))
            new_data_list.append(h5py.File(msvd_p + msvd_obj[1], 'r'))
            new_data_list.append(h5py.File(msvd_p + msvd_obj[2], 'r'))
        else:
            new_data_list.append(h5py.File(msrvtt_p + msrvtt_obj[0], 'r'))
            new_data_list.append(h5py.File(msrvtt_p + msrvtt_obj[1], 'r'))
            new_data_list.append(h5py.File(msrvtt_p + msrvtt_obj[2], 'r'))

        print("特征向量组合中")
        with h5py.File(h5file_save_path, 'w') as h5py_file:
            video_full_fts = h5py_file.create_group("video_full_fts")
            video_fts = h5py_file.create_group("video_fts")
            video_kfs = h5py_file.create_group("video_kfs")
            video_obj_fts = h5py_file.create_group("video_obj_fts")
            full_paths, file_names, base_names = File_Tools.get_filenames(video_path)
            for i in tqdm(range(len(full_paths))):
                full_images_fts = old_data["video_full_fts"][str(base_names[i])][()]
                image_fts = old_data["video_fts"][str(base_names[i])][()]
                selected_frames_indices = old_data["video_kfs"][str(base_names[i])][()]
                for obj_f in new_data_list:
                    try:
                        obj_fts = obj_f[str(base_names[i])]['feats'][()]
                        print(obj_fts.shape)
                        break
                    except:
                        print(base_names[i] + "数据不存在")

                video_fts.create_dataset(str(base_names[i]), data=image_fts)
                video_kfs.create_dataset(str(base_names[i]), data=selected_frames_indices)
                video_full_fts.create_dataset(str(base_names[i]), data=full_images_fts)
                video_obj_fts.create_dataset(str(base_names[i]), data=obj_fts)
            h5py_file.close()
        print("特征文件已生成")

    @staticmethod
    def get_images_by_h5(fts_path, videos_path, output_path):
        """
        根据特征文件中的关键帧信息提取视频文件中的对应帧并保存为jpg格式
        input：keyframe[]
        output: imagedatasets
        """
        full_paths, file_names, base_names = File_Tools.get_filenames(videos_path)
        fts_data = h5py.File(fts_path, 'r')
        for k in range(len(full_paths)):
            selected_frames_indices = fts_data["video_kfs"][str(base_names[k])][()]
            video_path = full_paths[k]
            base_name = base_names[k]
            print("==============================" + str(k) + "========================================")
            print(selected_frames_indices)
            cap = cv2.VideoCapture(video_path)
            FPS = cap.get(cv2.CAP_PROP_FPS)
            sample_time = FPS // 3

            i = 0
            j = 0
            while (cap.isOpened()):
                ret, cv2_im = cap.read()
                if ret and i % sample_time == 0:
                    if not os.path.exists(output_path + base_name):
                        os.makedirs(output_path + base_name)
                    img_output_path = output_path + base_name + f"/frame_{j}.jpg"  # 保存帧图片的路径和文件名
                    if j in selected_frames_indices:
                        cv2.imwrite(img_output_path, cv2_im)  # 保存帧图片
                        print(img_output_path)
                    j += 1
                elif not ret:
                    break

                i += 1

            cap.release()

if __name__ == '__main__':
    # videoUtils = VideoUtils()

    # image_fts, selected_frames_indices, frames_fts = videoUtils.get_clip_video_feats("/home/wy3/zjc_data/datasets/MSVD-QA/video/_ZwwKOzpt2I_69_76.avi")
    # print(frames_fts.shape)

    # videoUtils.create_feature_hdf5("/home/wy3/zjc_data/datasets/MSVD-QA/video/",
    #                                "/home/wy3/zjc_data/datasets/MSVD-QA/" + 'msvd_video_clip_RN50X60_fts.hdf5')

    # videoUtils.add_obj_feature_hdf5("/home/wy3/zjc_data/datasets/MSVD-QA/video/",
    #                                "/home/wy3/zjc_data/datasets/MSVD-QA/" + 'msvd_video_clip_RN50X60_fts.hdf5', "/home/wy3/zjc_data/datasets/MSVD-QA/" + 'msvd_video_clip_RN50X60_fts_four.hdf5')


    # videoUtils.create_feature_hdf5("/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/", "/home/no-zjc/datasets/MSR-VTT/" + 'msr-vtt_clip_RN50X60_fts.hdf5')

    # videoUtils.add_obj_feature_hdf5("/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/",
    #                                 "/home/wy3/zjc_data/datasets/MSR-VTT/" + 'msr-vtt_clip_RN50X60_fts.hdf5',
    #                                 "/home/wy3/zjc_data/datasets/MSR-VTT/" + 'msr-vtt_clip_RN50X60_fts_four.hdf5')



    # videoUtils.create_feature_hdf5("/home/wy3/zjc_data/datasets/MSVD-QA/video/",
    #                                "/home/wy3/zjc_data/datasets/MSVD-QA/" + 'msvd_video_clip_ViT_14_fts.hdf5')

    # videoUtils.add_obj_feature_hdf5("/home/wy3/zjc_data/datasets/MSVD-QA/video/",
    #                                 "/home/wy3/zjc_data/datasets/MSVD-QA/" + 'msvd_video_clip_ViT_L_14_fts.hdf5',
    #                                 "/home/wy3/zjc_data/datasets/MSVD-QA/" + 'msvd_video_clip_ViT_L_14_fts_four.hdf5')

    # videoUtils.create_feature_hdf5("/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/",
    #                                "/home/no-zjc/datasets/MSR-VTT/" + 'msr-vtt_clip_ViT_14_fts.hdf5')

    # videoUtils.add_obj_feature_hdf5("/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/",
    #                                 "/home/wy3/zjc_data/datasets/MSR-VTT/" + 'msr-vtt_clip_ViT_L_14_fts.hdf5',
    #                                 "/home/wy3/zjc_data/datasets/MSR-VTT/" + 'msr-vtt_clip_ViT_L_14_fts_four.hdf5')

    # videoUtils.create_feature_hdf5("/home/wy3/zjc_data/datasets/MSR-VTT/data/test-video/", "/home/no-zjc/datasets/MSR-VTT/" + 'msr-vtt_test_clip_fts.hdf5')

    # VideoUtils.get_images_by_h5("/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_clip_ViT_L_14_fts_four.hdf5", "/home/wy3/zjc_data/datasets/MSVD-QA/video/", "/home/wy3/zjc_data/datasets/MSVD-QA/key_images/")
    VideoUtils.get_images_by_h5("/home/wy3/zjc_data/datasets/MSR-VTT/" + 'msr-vtt_clip_ViT_L_14_fts_four.hdf5', "/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/", "/home/wy3/zjc_data/datasets/MSR-VTT/key_images/")










