# coding=utf-8
"""
    @project: few-shot-video-to-text
    @Author：no-zjc
    @file： Hallucination_identity.py
    @date：2023/12/30 6:51
"""
import json
import random

import clip
import h5py
from tqdm import tqdm

from VideoCaption_Hallucination.model.dataset_create.hu_dataset_tool import load_data_for_training_HU
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from VideoCaption_Hallucination.File_Tools import File_Tools

dt = 1
gen_type = "001"
dataset_type = ['msvd', 'msrvtt']
data_fts_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_clip_ViT_L_14_fts_four.hdf5",
                 "/home/wy3/zjc_data/datasets/MSR-VTT/msr-vtt_clip_ViT_L_14_fts_four.hdf5"]
annotations_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/msvd_hallucination_dataset_sv.json",
                    "/home/wy3/zjc_data/datasets/MSR-VTT/msrvtt_hallucination_dataset_sv.json"]

model_name = dataset_type[dt] + '_hallucination_identity_' + gen_type + '.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, feature_extractor = clip.load("ViT-L/14", device=device, download_root='../../../clip_checkpoints')
num_epochs = 20
batch_size = 10


def get_features_labels(split_text, clip_model, a_path, fts_path):
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


def evaluate_model(test_hu_features, test_hu_labels, eval_model):
    test_features = torch.tensor(test_hu_features).to(torch.float32)
    test_labels = torch.tensor(test_hu_labels).to(torch.float32)

    result_dict = []
    label_dict = test_labels.cpu().tolist()
    for i in tqdm(range(0, test_features.size(0), batch_size)):
        test_inputs = test_features[i:i + batch_size].to(device)
        test_targets = test_labels[i:i + batch_size].to(device)
        # 进行预测
        with torch.no_grad():
            eval_model.eval()
            test_outputs = eval_model(test_inputs)
            outputs = test_outputs.cpu().tolist()
            for it in outputs:
                result_dict.append(it)

    test_num = len(result_dict)
    test_truth = 0
    test_no_hu = 0
    test_obj_hu = 0
    test_act_hu = 0
    num_no_hu = 0
    num_obj_hu = 0
    num_act_hu = 0
    for i in range(len(result_dict)):
        gt = label_dict[i].index(max(label_dict[i]))
        pr = result_dict[i].index(max(result_dict[i]))
        if gt == 0:
            num_no_hu = num_no_hu + 1
        if gt == 1:
            num_obj_hu = num_obj_hu + 1
        if gt == 2:
            num_act_hu = num_act_hu + 1
        if gt == pr:
            test_truth = test_truth + 1
            if gt == 0:
                test_no_hu = test_no_hu + 1
            if gt == 1:
                test_obj_hu = test_obj_hu + 1
            if gt == 2:
                test_act_hu = test_act_hu + 1

    acc = test_truth / test_num
    no_acc = test_no_hu / num_no_hu
    obj_acc = test_obj_hu / num_obj_hu
    act_acc = test_act_hu / num_act_hu
    acc_dict = {"acc": acc, "no_acc":  no_acc, "obj_acc":  obj_acc, "act_acc":  act_acc}
    print("当前epoch的准确率为：" + json.dumps(acc_dict))
    return acc_dict


class HallucinationIdentityModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(HallucinationIdentityModel, self).__init__()
        self.transformer_layer1 = nn.TransformerEncoderLayer(input_size, nhead=8)
        self.transformer_layer2 = nn.TransformerEncoderLayer(input_size, nhead=8)
        self.layerNorm = nn.LayerNorm(input_size)
        self.mlp_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layerNorm(x)
        x = self.transformer_layer1(x)
        x = self.transformer_layer2(x)
        x = torch.mean(x, dim=1)
        x = self.layerNorm(x)
        x = self.mlp_layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def train_model():
    train_hu_data, train_hu_features, train_hu_labels = get_features_labels("train", clip_model, annotations_path[dt], data_fts_path[dt])
    val_hu_data, val_hu_features, val_hu_labels = get_features_labels("val", clip_model, annotations_path[dt], data_fts_path[dt])

    # 转换为PyTorch Tensor
    features = torch.tensor(train_hu_features).to(torch.float32)
    labels = torch.tensor(train_hu_labels).to(torch.float32)

    # 创建模型实例
    model = HallucinationIdentityModel(input_size=768, hidden_size=256, num_classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    best_acc = 0
    for epoch in range(num_epochs):
        # 打乱数据
        indices = torch.randperm(features.size(0))
        features = features[indices]
        labels = labels[indices]

        # 将数据划分为小批量(batch)
        for i in tqdm(range(0, features.size(0), batch_size)):
            inputs = features[i:i + batch_size].to(device)
            targets = labels[i:i + batch_size].to(device)
            # 前向传播
            outputs = model(inputs)

            # 计算损失函数
            loss = criterion(outputs, targets)
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_model(val_hu_features, val_hu_labels, model)
        if acc["acc"] > best_acc:
            best_acc = acc["acc"]
            # 保存训练好的模型
            torch.save(model.state_dict(), model_name)

        print('Epoch [{}/{}], Loss: {:.4f}, acc: {}, best_acc: {}'.format(epoch + 1, num_epochs, loss.item(), acc["acc"],
              best_acc))

    File_Tools.write_to_json(dataset_type[0] + '_hallucination_identity_' + gen_type + '.json', {"model_name": model_name, "acc": best_acc})

if __name__ == '__main__':

    # train_model()

    # 创建模型实例
    eval_model = HallucinationIdentityModel(input_size=768, hidden_size=256, num_classes=3)

    # 加载保存的模型参数
    eval_model.load_state_dict(torch.load(model_name))

    # eval_model = model
    eval_model.to(device)

    test_hu_data, test_hu_features, test_hu_labels = get_features_labels("test", clip_model, annotations_path[dt], data_fts_path[dt])
    evaluate_model(test_hu_features, test_hu_labels, eval_model)
