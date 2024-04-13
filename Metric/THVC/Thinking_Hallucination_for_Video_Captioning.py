import os
import sys

# import pickle5 as pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import warnings

import VideoCaption_Hallucination.Attention_Visualization_Tool
import zero_shot_video_caption_Hallucination


# Import Path,Vocabulary, utility, evaluator and datahandler module
from Metric.THVC.config import Path
from Metric.THVC.dictionary import Vocabulary
from Metric.THVC.utils import Utils
from Metric.THVC.evaluate import Evaluator
from Metric.THVC.data import DataHandler
from Metric.THVC.coaha import COAHA

import random
import numpy as np
import copy
from tqdm import tqdm


# Import configuration and model
from Metric.THVC.config import ConfigTHVC
from Metric.THVC.models.THVC.model import THVC
from Metric.coco_caption.my_eval.eval_script import coco_evaluation_by_json
from VideoCaption_Hallucination.File_Tools import File_Tools
from VideoCaption_Hallucination.model.infer import *
from VideoCaption_Hallucination import Attention_Visualization_Tool


def evaluation_coaha(predict, dataset_str):
    """
    :param predict: 需要评估的预测字幕文件：视频：描述  可以使用zero_shot_video_caption_Hallucination.msvd_caption_result_mapping函数进行格式化
    :param dataset_str: 数据集字符串 如 msvd
    :return: 测试的coaha值
    """
    # cfg.dataset = dataset_str

    warnings.filterwarnings('ignore')
    # set seed for reproducibility
    utils = Utils()
    utils.set_seed(1)

    cfg = ConfigTHVC()
    cfg.dataset = 'msvd'
    cfg.vocabulary_min_count = 5
    cfg.create_entity = False

    # creation of path object
    path = Path(cfg, os.getcwd() + "/Metric/THVC")
    # Vocabulary object
    voc = Vocabulary(cfg)

    voc.load()
    # remove all words below count min_count
    min_count = cfg.vocabulary_min_count
    voc.trim(min_count=min_count)
    print('Vocabulary Size : ', voc.num_words)

    # Datasets and dataloaders
    data_handler = DataHandler(cfg, path, voc)
    train_dset, val_dset, test_dset = data_handler.getDatasets()
    train_loader, val_loader, test_loader = data_handler.getDataloader(train_dset, val_dset, test_dset)
    if cfg.opt_auxiliary_heads:
        cfg.update_head_info(data_handler.auxhead_data)

    # COAHA evaluator
    coaha_test = COAHA(cfg, data_handler.test_dict)

    coaha_test.evaluate(predict)
    return coaha_test.coaha_total

def evaluate_coaha_metric():
    warnings.filterwarnings('ignore')
    # set seed for reproducibility
    utils = Utils()
    utils.set_seed(1)

    cfg = ConfigTHVC()

    checkpoint_path = "MSVD_norag_7M_loc_gpt2_40_01_13_22_50/checkpoint-7650"
    epoch = 10

    root_path = "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result/"
    json_path = root_path + checkpoint_path
    if "msvd" in json_path or "MSVD" in json_path:
        infer_msvd(json_path, disable_rag=True)
        result_text, evaluate_result_dict = coco_evaluation_by_json(json_path + "/MSVD_val_preds.json",
                                                                    "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Metric/coco_caption/my_eval/data/lable_references/msvd_video_caption_format.json",
                                                                    epoch)
        pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_format(json_path + "/MSVD_val_preds.json")
        cfg.dataset = 'msvd'
    else:
        infer_msrvtt(json_path, disable_rag=True)
        result_text, evaluate_result_dict = coco_evaluation_by_json(json_path + "/MSR-VTT_val_preds.json",
                                                                    "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Metric/coco_caption/my_eval/data/lable_references/msrvtt_caption.json",
                                                                    epoch)
        pre_dict = zero_shot_video_caption_Hallucination.msrvtt_caption_result_format(
            json_path + "/MSR-VTT_val_preds.json")
        cfg.dataset = 'msrvtt'

    cfg.vocabulary_min_count = 5
    cfg.create_entity = False

    # creation of path object
    path = Path(cfg, os.getcwd())
    # Vocabulary object
    voc = Vocabulary(cfg)
    # If vocabulary is already saved or downloaded the saved file
    # comment this if using vocabulary for the first time or with no saved file
    voc.load()
    # remove all words below count min_count
    min_count = cfg.vocabulary_min_count
    voc.trim(min_count=min_count)
    print('Vocabulary Size : ', voc.num_words)

    # Datasets and dataloaders
    data_handler = DataHandler(cfg, path, voc)
    train_dset, val_dset, test_dset = data_handler.getDatasets()
    train_loader, val_loader, test_loader = data_handler.getDataloader(train_dset, val_dset, test_dset)
    if cfg.opt_auxiliary_heads:
        cfg.update_head_info(data_handler.auxhead_data)

    # Model object
    model = THVC(voc, cfg, path)
    # Standard Evaluator
    # Evaluator object on test data
    # test_evaluator_greedy = Evaluator(model, test_loader, path, cfg, data_handler.test_dict)
    # test_evaluator_beam = Evaluator(model,test_loader,path,cfg,data_handler.test_dict,decoding_type='beam')

    # COAHA evaluator

    eval_data = {}
    # eval_data.update(data_handler.train_dict)
    # eval_data.update(data_handler.val_dict)
    eval_data.update(data_handler.test_dict)
    # coaha_test = COAHA(cfg, data_handler.test_dict)
    coaha_test = COAHA(cfg, eval_data)
    """

    #Training Loop
    cfg.encoder_lr = 1e-4
    cfg.decoder_lr = 1e-4
    cfg.teacher_forcing_ratio = 1.0

    model.update_hyperparameters(cfg)
    coaha = []

    #lr_scheduler = StepLR(model.dec_optimizer,300,gamma=0.1,verbose=False)
    # lr_scheduler = ReduceLROnPlateau(model.dec_optimizer, mode='min', factor=cfg.lr_decay_gamma,
    #                                      patience=cfg.lr_decay_patience, verbose=True)
    for e in range(1,901):
        loss_train,ac_loss = model.train_epoch(train_loader,utils)
        #lr_scheduler.step()
        if e%25 == 0 :
            print('Epoch -- >',e,'Loss -->',loss_train,'  AC loss --->',ac_loss)
            print('greedy :',test_evaluator_greedy.evaluate(utils,model,e,loss_train))
            #coaha_test.evaluate(test_evaluator_greedy.prediction_dict)
            #print('COAHA SCORE: ',coaha_test.coaha_total)
            #coaha.append(coaha_test.coaha_total)


    """

    # pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_mapping("../../result/youtube_mapping.txt",
    #                                                                              "../../result/zero_shot_video_caption_all_pre.txt",
    #                                                                              "1201", "")

    # pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_mapping("../../result/youtube_mapping.txt",
    #                                                                              "../coco_caption/my_eval/data/caption_output/10_MSVD_val_preds.txt",
    #                                                                              "1201", "",
    #                                                                              )

    # pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_format("../coco_caption/my_eval/data/caption_output/10_MSVD_val_preds.json")
    coaha_test.evaluate(pre_dict)
    print(coaha_test.coaha_total)
    evaluate_result_dict.update({"COAHA": coaha_test.coaha_total})
    result_text = result_text + "COAHA" + " : " + str(coaha_test.coaha_total)
    print(evaluate_result_dict)
    print(result_text)
    File_Tools.write_to_txt(json_path + "/evaluate_result.txt", result_text)
    # model.load_state_dict(torch.load('saved_model.pt'))
    # model.eval()
    # print(test_evaluator_greedy.evaluate(utils, model, 900, 999))
    # print(test_evaluator_greedy.prediction_dict)
    #
    # coaha_test.evaluate(test_evaluator_greedy.prediction_dict)
    # print(coaha_test.coaha_total)

def create_coaha_test(dataset_name):
    warnings.filterwarnings('ignore')
    # set seed for reproducibility
    utils = Utils()
    utils.set_seed(1)
    cfg = ConfigTHVC()
    cfg.dataset = dataset_name

    cfg.vocabulary_min_count = 5
    cfg.create_entity = False

    # creation of path object
    path = Path(cfg, os.getcwd())
    # Vocabulary object
    voc = Vocabulary(cfg)
    # If vocabulary is already saved or downloaded the saved file
    # comment this if using vocabulary for the first time or with no saved file
    voc.load()
    # remove all words below count min_count
    min_count = cfg.vocabulary_min_count
    voc.trim(min_count=min_count)
    print('Vocabulary Size : ', voc.num_words)

    # Datasets and dataloaders
    data_handler = DataHandler(cfg, path, voc)
    train_dset, val_dset, test_dset = data_handler.getDatasets()
    train_loader, val_loader, test_loader = data_handler.getDataloader(train_dset, val_dset, test_dset)
    if cfg.opt_auxiliary_heads:
        cfg.update_head_info(data_handler.auxhead_data)

    eval_data = {}
    # eval_data.update(data_handler.train_dict)
    # eval_data.update(data_handler.val_dict)
    eval_data.update(data_handler.test_dict)
    coaha_test = COAHA(cfg, eval_data)

    # dataset_path = ["/home/wy3/zjc_data/datasets/MSVD-QA/action_object/", "/home/wy3/zjc_data/datasets/MSR-VTT/action_object/"]
    # if dataset_name == "msvd":
    #     dp = dataset_path[0]
    # else:
    #     dp = dataset_path[1]
    #
    # File_Tools.write_to_json(dp + dataset_name + "_action_add.json", coaha_test.action_add)
    # File_Tools.write_to_json(dp + dataset_name + "_action_list.json", coaha_test.action_list)
    # File_Tools.write_to_json(dp + dataset_name + "_action_dict.json", coaha_test.action_dict)
    # File_Tools.write_to_json(dp + dataset_name + "_object_list.json", coaha_test.object_list)
    # File_Tools.write_to_json(dp + dataset_name + "_object_dict.json", coaha_test.object_dict)

    return coaha_test

def evaluate_metric_one(checkpoint_path, epoch, disable_rag, coaha_test, result_file_label=""):
    warnings.filterwarnings('ignore')
    execution_time_str = ""
    if "msvd" in checkpoint_path or "MSVD" in checkpoint_path:
        if os.path.exists(checkpoint_path + "/MSVD_val_preds" + result_file_label + ".json"):
            print("检测到当前模型已经存在测试结果文件！路径为：" + checkpoint_path + "/MSVD_val_preds" + result_file_label + ".json")
            Attention_Visualization_Tool.attention_write_label = 0
        else:
            execution_time_str = infer_msvd(checkpoint_path, "MSVD_val_preds" + result_file_label + ".json", disable_rag)
        # execution_time_str = infer_msvd(checkpoint_path, "MSVD_val_preds" + result_file_label + ".json", disable_rag)

        result_text, evaluate_result_dict = coco_evaluation_by_json(checkpoint_path + "/MSVD_val_preds" + result_file_label + ".json",
                                                                    "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Metric/coco_caption/my_eval/data/lable_references/msvd_video_caption_format.json",
                                                                    epoch)
        pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_format(checkpoint_path + "/MSVD_val_preds" + result_file_label + ".json")
    else:
        if os.path.exists(checkpoint_path + "/MSR-VTT_val_preds" + result_file_label + ".json"):
            print("检测到当前模型已经存在测试结果文件！路径为：" + checkpoint_path + "/MSR-VTT_val_preds" + result_file_label + ".json")
            Attention_Visualization_Tool.attention_write_label = 0
        else:
            execution_time_str = infer_msrvtt(checkpoint_path, "MSR-VTT_val_preds" + result_file_label + ".json", disable_rag)
        # execution_time_str = infer_msrvtt(checkpoint_path, "MSR-VTT_val_preds" + result_file_label + ".json", disable_rag)

        result_text, evaluate_result_dict = coco_evaluation_by_json(checkpoint_path + "/MSR-VTT_val_preds" + result_file_label + ".json",
                                                                    "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Metric/coco_caption/my_eval/data/lable_references/msrvtt_caption.json",
                                                                    epoch)
        pre_dict = zero_shot_video_caption_Hallucination.msrvtt_caption_result_format(checkpoint_path + "/MSR-VTT_val_preds" + result_file_label + ".json")

    if execution_time_str != "":
        evaluate_result_dict.update({"infer_execution_time": execution_time_str})
        result_text = result_text + "infer_execution_time" + " : " + str(execution_time_str) + "\n"

    if Attention_Visualization_Tool.attention_write_label == 1:
        FT.write_to_json(checkpoint_path + "/evaluate_result" + result_file_label + "_attention_visualization_data.json", Attention_Visualization_Tool.attention_write_all_list)

    coaha_test.evaluate(pre_dict)

    if coaha_test.hallucinated_video_info is not None:
        FT.write_to_json(
            checkpoint_path + "/evaluate_result" + result_file_label + "_coaha_info.json",
            coaha_test.hallucinated_video_info)


    if Attention_Visualization_Tool.coaha_word_test_by_result == 1:
        pass
    evaluate_result_dict.update({"OBJ_H": coaha_test.oh_total})
    evaluate_result_dict.update({"ACT_H": coaha_test.ah_total})
    evaluate_result_dict.update({"COAHA": coaha_test.coaha_total})
    result_text = result_text + "OBJ_H" + " : " + str(coaha_test.oh_total) + "\n"
    result_text = result_text + "ACT_H" + " : " + str(coaha_test.ah_total) + "\n"
    result_text = result_text + "COAHA" + " : " + str(coaha_test.coaha_total) + "\n"
    print(evaluate_result_dict)
    print(result_text)
    File_Tools.write_to_txt(checkpoint_path + "/evaluate_result" + result_file_label + ".txt", result_text)
    print("结果文件已保存到")

def evaluate_more(result_path, epoch_batch_num, start_epoch, stride, epoch_num, dataset_name, result_file_label=""):
    root_path = "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result/"
    # print("---------coaha test tool create--------")
    coaha_test = create_coaha_test(dataset_name)
    # coaha_test = None
    for epoch in range(start_epoch, epoch_num+1, stride):
        model_path = "/checkpoint-" + str(epoch_batch_num * epoch)

        checkpoint_path = root_path + result_path + model_path
        disable_rag = True
        if "_rag_" in result_path:
            disable_rag = False
        print("===========================evaluating model " + checkpoint_path + "================================")
        # if os.path.exists(checkpoint_path + "/evaluate_result" + result_file_label + ".txt") and not re_evaluate:
        #     print("检测到当前模型已存在评估结果文件，请删除后再试，当前默认输出原评估结果！路径为：" + checkpoint_path + "/evaluate_result.txt")
        #     ev = open(checkpoint_path + "/evaluate_result.txt", 'r').read()
        #     print(ev)
        #     continue
        evaluate_metric_one(checkpoint_path, epoch, disable_rag, coaha_test, result_file_label)

def format_Contrast_model_result_evaluate(result_path, model_name, output_file, dataset_name):
    result_data = File_Tools.load_json_data(result_path)
    formate_data = []
    mapping_dict = File_Tools.get_msvd_mapping_dict()
    if model_name == "HMN" and not os.path.exists(output_file):
        for key, value in result_data.items():
            if dataset_name == "msvd":
                video_id = int(mapping_dict[key].split("vid")[1])
            else:
                video_id = int(key.split("video")[1])
            formate_data.append({"file_name": key, "caption": value, "video_id": video_id})

        File_Tools.write_to_json(output_file, formate_data)

    epoch = 0
    warnings.filterwarnings('ignore')
    coaha_test = create_coaha_test(dataset_name)
    if "msvd" in dataset_name or "MSVD" in dataset_name:
        result_text, evaluate_result_dict = coco_evaluation_by_json(output_file,
                                                                    "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Metric/coco_caption/my_eval/data/lable_references/msvd_video_caption_format.json",
                                                                    epoch)
        pre_dict = zero_shot_video_caption_Hallucination.msvd_caption_result_format(
            output_file)
    else:
        result_text, evaluate_result_dict = coco_evaluation_by_json(output_file,
                                                                    "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Metric/coco_caption/my_eval/data/lable_references/msrvtt_caption.json",
                                                                    epoch)
        pre_dict = zero_shot_video_caption_Hallucination.msrvtt_caption_result_format(
            output_file)

    coaha_test.evaluate(pre_dict)
    evaluate_result_dict.update({"OBJ_H": coaha_test.oh_total})
    evaluate_result_dict.update({"ACT_H": coaha_test.ah_total})
    evaluate_result_dict.update({"COAHA": coaha_test.coaha_total})
    result_text = result_text + "OBJ_H" + " : " + str(coaha_test.oh_total) + "\n"
    result_text = result_text + "ACT_H" + " : " + str(coaha_test.ah_total) + "\n"
    result_text = result_text + "COAHA" + " : " + str(coaha_test.coaha_total) + "\n"
    print(evaluate_result_dict)
    print(result_text)
    File_Tools.write_to_txt(output_file + "_evaluate_result.txt", result_text)

def THVC_method():
    # Import configuration and model
    from config import ConfigTHVC
    from models.THVC.model import THVC

    # set seed for reproducibility
    utils = Utils()
    utils.set_seed(1)

    cfg = ConfigTHVC()

    # Change of Hyperparameters
    cfg.dataset = 'msvd'
    cfg.vocabulary_min_count = 5
    cfg.create_entity = False

    # creation of path object
    path = Path(cfg, os.getcwd())
    # Vocabulary object
    voc = Vocabulary(cfg)
    # If vocabulary is already saved or downloaded the saved file
    voc.load()  # comment this if using vocabulary for the first time or with no saved file
    min_count = cfg.vocabulary_min_count  # remove all words below count min_count
    voc.trim(min_count=min_count)
    print('Vocabulary Size : ', voc.num_words)
    # Datasets and dataloaders
    data_handler = DataHandler(cfg, path, voc)
    train_dset, val_dset, test_dset = data_handler.getDatasets()
    train_loader, val_loader, test_loader = data_handler.getDataloader(train_dset, val_dset, test_dset)
    if cfg.opt_auxiliary_heads:
        cfg.update_head_info(data_handler.auxhead_data)

    # Model object
    model = THVC(voc, cfg, path)
    # Standard Evaluator
    # Evaluator object on test data
    test_evaluator_greedy = Evaluator(model, test_loader, path, cfg, data_handler.test_dict)
    # test_evaluator_beam = Evaluator(model,test_loader,path,cfg,data_handler.test_dict,decoding_type='beam')

    # COAHA evaluator
    # coaha_test = COAHA(cfg, data_handler.test_dict)

    model.load_state_dict(torch.load('saved_model.pt'))
    model.eval()
    test_evaluator_greedy.evaluate(utils, model, 900, 999)

    # coaha_test.evaluate(test_evaluator_greedy.prediction_dict)
    # coaha_test.coaha_total

if __name__ == '__main__':
    root_path = "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result/"
    # result_path = "old/MSVD_norag_7M_loc_gpt2_40_t01_14_17_10_bs8"
    result_path = "MSVD_rag_7M_loc_gpt2_20_t01_24_14_36_bs8"
    dataset_name = "msrvtt"
    re_evaluate = True
    if "msvd" in result_path or "MSVD" in result_path:
        dataset_name = "msvd"
    epoch_batch_num = 15250
    start_epoch = 1
    stride = 1
    epoch_num = 1
    # 测试标记当前结果的类别，仅影响结果文件命名，可自定义
    result_file_label = ""

    print(result_file_label)
    evaluate_more(result_path, epoch_batch_num, start_epoch, stride, epoch_num, dataset_name, result_file_label)



    # if Attention_Visualization_Tool.attention_write_label == 1:
    #     FT.write_to_json(root_path + result_path + "/" + dataset_name + result_file_label + "_attention_visualization_data.json", Attention_Visualization_Tool.attention_write_all_list)
    # THVC_method()

    # contrast_file = "/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Contrast_Model/"
    # HMN_result_path = ["HMN/HMN_MSVD.json", "HMN/HMN_MSRVTT.json"]
    # format_Contrast_model_result_evaluate(result_path=contrast_file + HMN_result_path[0], model_name="HMN", output_file="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Contrast_Model/HMN/HMN_MSVD_format.json", dataset_name="msvd")
    # format_Contrast_model_result_evaluate(result_path=contrast_file + HMN_result_path[1], model_name="HMN",
    #                                       output_file="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/Contrast_Model/HMN/HMN_MSRVTT_format.json",
    #                                       dataset_name="msrvtt")