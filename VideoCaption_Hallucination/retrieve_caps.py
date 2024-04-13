import json

import h5py
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# msvd
m_mapping_path = "/home/wy3/zjc_data/datasets/MSVD-QA/youtube_mapping.txt"
m_fts_path = "/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_random_clip_fts.hdf5"
m_path = "/home/wy3/zjc_data/datasets/MSVD-QA/"

# msr-vtt
mv_annot_path = "/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json"
mv_fts_path = "/home/wy3/zjc_data/datasets/MSR-VTT/msr-vtt_train_clip_fts.hdf5"
mv_path = "/home/wy3/zjc_data/datasets/MSR-VTT/"

def load_coco_data(coco_data_path):
    """We load in all images and only the train captions."""

    annotations = json.load(open(coco_data_path))['images']
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'restval':
            item['split'] = 'train'
        if item['split'] == 'train':
            for sentence in item['sentences']:
                captions.append({'image_id': item['cocoid'], 'caption': ' '.join(sentence['tokens'])})
        images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})

    return images, captions


def load_msvd_data(mapping_path):
    videos = []
    with open(mapping_path, 'r') as file:
        for line in file:
            words = line.split()

            file_name = words[0] + ".avi"
            vid = words[1].split("\n")[0]
            vid_int = int(vid.split("vid")[1])

            videos.append({'file_name': file_name, 'vid': vid_int, 'fts_key': words[0]})

    return videos

def load_msr_vtt_data(annot_path):
    annotations = json.load(open(annot_path))
    videos = []
    for i, (key, value) in enumerate(annotations.items()):
        file_name = key + ".mp4"
        vid_int = int(key.split("video")[1])
        videos.append({'file_name': file_name, 'vid': vid_int, 'fts_key': key})
    return videos

def filter_captions(data):
    decoder_name = '../loc_gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    bs = 1

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(caps[idx:idx + bs], return_tensors='np')['input_ids'].tolist()

    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions


def encode_captions(captions, model, device):
    # with h5py.File('fts_files/encoded_captions_feature.h5', 'r') as f:
    #     encoded_captions = f['encoded_captions'][:]

    bs = 1
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx+bs]).to(device)
            caption_clip = model.encode_text(input_ids)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    # with h5py.File('fts_files/encoded_captions_feature.h5', 'w') as f:
    #     f.create_dataset('encoded_captions', data=encoded_captions)

    return encoded_captions


def encode_images(images, image_path, model, feature_extractor, device):
    image_ids = [i['image_id'] for i in images]

    bs = 64
    image_features = []

    for idx in tqdm(range(0, len(images), bs)):
        image_input = [feature_extractor(Image.open(os.path.join(image_path, i['file_name'])))
                       for i in images[idx:idx + bs]]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)

    return image_ids, image_features


def encode_videos(videos, video_features_path, dataset_str):

    video_features = h5py.File(video_features_path, 'r')
    fin_video_features = []
    video_ids = []
    video_fts_keys = []
    if dataset_str == "msvd":
        for i in videos:
            video_ids.append(i["vid"])
            video_fts_keys.append(i["fts_key"])

        # 视频特征处理
        for idx in range(len(video_ids)):
            encoder_outputs = video_features["video_fts"][video_fts_keys[idx]][()]
            pooled_features = np.mean(encoder_outputs, axis=0)
            fin_video_features.append(pooled_features)

        fin_video_features = np.stack(fin_video_features)

    if dataset_str == "msr-vtt":
        for i in videos:
            video_ids.append(i["vid"])
            video_fts_keys.append(i["fts_key"])

        # 视频特征处理
        for idx in range(len(video_ids)):
            encoder_outputs = video_features["video_fts"][video_fts_keys[idx]][()]
            pooled_features = np.mean(encoder_outputs, axis=0)
            fin_video_features.append(pooled_features)

        fin_video_features = np.stack(fin_video_features)
    return video_ids, fin_video_features


def encode_videos_by_frame(videos, video_features_path, dataset_str):

    video_features = h5py.File(video_features_path, 'r')
    fin_video_features = []
    video_ids = []
    video_frama_ids = []
    video_fts_keys = []
    if dataset_str == "msvd":
        for i in videos:
            video_ids.append(i["vid"])
            video_fts_keys.append(i["fts_key"])

        # 视频特征处理 帧检索
        for idx in range(len(video_ids)):
            key_frames = video_features["video_kfs"][video_fts_keys[idx]][()]
            video_full_fts = video_features["video_full_fts"][video_fts_keys[idx]][()]
            for kf in key_frames:
                video_frama_ids.append(str(video_ids[idx]) + "_" + str(kf))
                frames_features = video_full_fts[kf]
                fin_video_features.append(frames_features)

        # 视频特征处理 平均池化
        # for idx in range(len(video_ids)):
        #     encoder_outputs = video_features["video_fts"][video_fts_keys[idx]][()]
        #     pooled_features = np.mean(encoder_outputs, axis=0)
        #     fin_video_features.append(pooled_features)

        fin_video_features = np.stack(fin_video_features)

    if dataset_str == "msr-vtt":
        for i in videos:
            video_ids.append(i["vid"])
            video_fts_keys.append(i["fts_key"])

        # 视频特征处理 帧检索
        for idx in range(len(video_ids)):
            key_frames = video_features["video_kfs"][video_fts_keys[idx]][()]
            video_full_fts = video_features["video_full_fts"][video_fts_keys[idx]][()]
            for kf in key_frames:
                video_frama_ids.append(str(video_ids[idx]) + "_" + str(kf))
                frames_features = video_full_fts[kf]
                fin_video_features.append(frames_features)
        # # 视频特征处理
        # for idx in range(len(video_ids)):
        #     encoder_outputs = video_features["video_fts"][video_fts_keys[idx]][()]
        #     pooled_features = np.mean(encoder_outputs, axis=0)
        #     fin_video_features.append(pooled_features)

        fin_video_features = np.stack(fin_video_features)
    return video_ids, video_frama_ids, fin_video_features


def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)

    return index, I


def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions


def combine_retrieved(video_ids, video_frama_ids, retrieved_caps):
    combine_retrieved_caps = {}
    video_id = 0
    video_all_captions = []
    video_captions = []
    for item in video_frama_ids:
        if item.split("_")[0] == str(video_ids[video_id]):
            frame_captions = retrieved_caps[item]
            video_all_captions.append(frame_captions)

        if item.split("_")[0] != str(video_ids[video_id]):
            for i in range(7):
                for fc in video_all_captions:
                    if fc not in video_captions:
                        video_captions.append(fc[i])
                if len(video_captions) >= 7:
                    break

            combine_retrieved_caps.update({video_ids[video_id]: video_captions})
            video_all_captions = []
            video_captions = []
            frame_captions = retrieved_caps[item]
            video_all_captions.append(frame_captions)
            video_id = video_id + 1

        if item == video_frama_ids[-1]:
            for i in range(7):
                for fc in video_all_captions:
                    if fc not in video_captions:
                        video_captions.append(fc[i])
                if len(video_captions) >= 7:
                    break

            combine_retrieved_caps.update({video_ids[video_id]: video_captions})
            video_captions = []

    return combine_retrieved_caps


def main():
    coco_data_path = '/home/wy3/zjc_data/ULFiles/dataset_coco.json'  # path to Karpathy splits downloaded from Kaggle
    # coco_data_path = 'data/data_test.json'  # path to Karpathy splits downloaded from Kaggle
    image_path = '../data/images/'
    dataset_type = ["msvd", "msr-vtt"]
    dataset_str = dataset_type[1]
    print('Loading data')
    images, captions = load_coco_data(coco_data_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, feature_extractor = clip.load("RN50x64", device=device)
    clip_model, feature_extractor = clip.load("ViT-L/14", device=device, download_root='../clip_checkpoints')

    print('Filtering captions')
    xb_image_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, device)

    print('Encoding images')
    if dataset_str == "msvd":
        # xq_image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)
        videos = load_msvd_data(m_mapping_path)
        # xq_image_ids, encoded_images = encode_videos(videos, m_fts_path, "msvd")
        video_ids, video_frama_ids, fin_video_features = encode_videos_by_frame(videos, m_fts_path, "msvd")
    if dataset_str == "msr-vtt":
        videos = load_msr_vtt_data(mv_annot_path)
        # xq_image_ids, encoded_images = encode_videos(videos, mv_fts_path, "msr-vtt")
        video_ids, video_frama_ids, fin_video_features = encode_videos_by_frame(videos, mv_fts_path, "msr-vtt")
    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, fin_video_features)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, video_frama_ids)


    print("combine frame retrieved captions")
    video_retrieved_caps = combine_retrieved(video_ids, video_frama_ids, retrieved_caps)

    print('Writing files')
    # faiss.write_index(index, "datastore/coco_index")
    # json.dump(captions, open('datastore/coco_index_captions.json', 'w'))




    if dataset_str == "msvd":
        json.dump(video_retrieved_caps, open(m_path + 'msvd_combine_frame_retrieved_caps_Vit_B32_2.json', 'w'))
    else:
        json.dump(video_retrieved_caps, open(mv_path + 'msrvtt_combine_frame_retrieved_caps_Vit_B32_2.json', 'w'))



if __name__ == '__main__':
    main()
