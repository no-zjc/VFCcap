import time

import clip
import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

import VideoCaption_Hallucination
from VideoCaption_Hallucination.File_Tools import File_Tools as FT
from VideoCaption_Hallucination.model.dataset_create.Hallucination_identity import HallucinationIdentityModel

from VideoCaption_Hallucination.video_utils import prep_strings, postprocess_preds
from VideoCaption_Hallucination.video_utils import VideoUtils
ImageFile.LOAD_TRUNCATED_IMAGES = True
from VideoCaption_Hallucination import Attention_Visualization_Tool


PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def evaluate_norag_model(args, videoUtils, feature_extractor, tokenizer, model, eval_df):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []
    bs = args.batch_size
    out_format_str = ""
    for idx in tqdm(range(0, len(eval_df), bs)):
        file_names = eval_df['file_name'][idx:idx+bs]
        video_ids = eval_df['vid'][idx:idx+bs]
        labels = eval_df['text'][idx:idx + bs]
        fts_keys = eval_df['fts_key'][idx:idx + bs]
        decoder_input_ids = [prep_strings('', tokenizer, is_test=True) for _ in range(len(video_ids))]
                
        # load video
        video_fts = []
        for file_name in file_names:

            image_fts, selected_frames_indices, full_images_fts = videoUtils.get_clip_video_feats(args.video_dir + file_name)
            input_tensor = torch.tensor(image_fts)

            input_tensor = input_tensor.to(torch.float32)

            # 定义填充量为负无穷，填充到 80x512 FloatTensor
            # padding = torch.tensor(float('-inf'))
            padding = torch.tensor(0)

            output_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, 80 - image_fts.shape[0]), value=padding)
            video_fts.append(output_tensor)

        # images = [Image.open(args.images_dir + file_name).convert("RGB") for file_name in file_names]
        # pixel_values = feature_extractor(images, return_tensors="pt").pixel_values

        encoder_outputs = torch.stack(video_fts, dim=0)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.to(args.device))

        with torch.no_grad():
            preds = model.generate(encoder_outputs=encoder_outputs,
                                   decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                                   **args.generation_kwargs)
            preds = tokenizer.batch_decode(preds)

        for video_id, pred, label, fn, fts_key in zip(video_ids, preds, labels, file_names, fts_keys):
            pred = postprocess_preds(pred, tokenizer)
            out.append({"video_id": int(video_id), "file_name": fts_key, "caption": pred, "label": label})

            # out_format_str += fts_key + "\t" + pred + "\n"
            # out_format_str += file_name + "\t" + pred + "\n"
    # FT.write_to_txt(args.txt_file, out_format_str)
    return out


def evaluate_model_bz_one(args, videoUtils, feature_extractor, tokenizer, model, eval_df):
    features_path = args.features_path
    template = open(args.template_path).read().strip() + ' '
    out = []
    out_format_str = ""
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        caps = eval_df['caps'][idx]
        video_id = eval_df['vid'][idx]
        label = eval_df['text'][idx]
        fts_key = eval_df['fts_key'][idx]
        if Attention_Visualization_Tool.attention_write_label == 1:
            Attention_Visualization_Tool.attention_write_fts_key = fts_key

        if args.disable_rag:
            decoder_input_ids = [prep_strings('', tokenizer, is_test=True)]
        else:
            decoder_input_ids = [prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                              k=int(args.k), is_test=True)]
        # load video
        if features_path is not None:
            features = h5py.File(features_path, 'r')
            encoder_outputs = VideoUtils.get_combine_features(features, fts_key)

        elif features_path is None:
            encoder_outputs = VideoUtils.get_combine_features("", fts_key, args.video_dir + file_name)

        encoder_outputs = torch.stack([encoder_outputs], dim=0)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.to(args.device))

        sg_info = FT.load_json_data(args.sg_path + fts_key + "/scene_graph_info.json")
        # 创建模型实例
        hallucination_model = HallucinationIdentityModel(input_size=768, hidden_size=256, num_classes=3)
        # 加载保存的模型参数
        hallucination_model.load_state_dict(torch.load(args.hallucination_model_path))
        # eval_model = model
        hallucination_model.to(args.device)

        clip_model, feature_extractor = clip.load("ViT-L/14", device=args.device)

        # with torch.no_grad():
        preds = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                               visual_constraints=sg_info,
                               hallucination_model=hallucination_model,
                               clip_model=clip_model,
                               **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)

        for pred in preds:
            pred = postprocess_preds(pred, tokenizer)
            out.append({"video_id": int(video_id), "file_name": fts_key, "caption": pred, "label": label, "RAG_caps": caps})
            # out_format_str += fts_key + "\t" + pred + "\n"

    # FT.write_to_txt(args.txt_file, out_format_str)
    return out

def evaluate_model_bz_one_baseline(args, videoUtils, feature_extractor, tokenizer, model, eval_df):
    features_path = args.features_path
    template = open(args.template_path).read().strip() + ' '
    out = []
    out_format_str = ""
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        caps = eval_df['caps'][idx]
        video_id = eval_df['vid'][idx]
        label = eval_df['text'][idx]
        fts_key = eval_df['fts_key'][idx]
        if args.disable_rag:
            decoder_input_ids = [prep_strings('', tokenizer, is_test=True)]
        else:
            decoder_input_ids = [prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                              k=int(args.k), is_test=True)]
        # load video
        if features_path is not None:
            features = h5py.File(features_path, 'r')
            encoder_outputs = VideoUtils.get_combine_features(features, fts_key)

        elif features_path is None:
            encoder_outputs = VideoUtils.get_combine_features("", fts_key, args.video_dir + file_name)

        encoder_outputs = torch.stack([encoder_outputs], dim=0)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.to(args.device))

        # with torch.no_grad():
        preds = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                               **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)

        for pred in preds:
            pred = postprocess_preds(pred, tokenizer)
            out.append({"video_id": int(video_id), "file_name": fts_key, "caption": pred, "label": label, "RAG_caps": caps})
            # out_format_str += fts_key + "\t" + pred + "\n"

    # FT.write_to_txt(args.txt_file, out_format_str)
    return out


def evaluate_rag_model(args, videoUtils, feature_extractor, tokenizer, model, eval_df):
    """RAG models can only be evaluated with a batch of length 1."""
    
    template = open(args.template_path).read().strip() + ' '

    # if args.features_path is not None:
    #     features = h5py.File(args.features_path, 'r')
    out = []
    out_format_str = ""
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        caps = eval_df['caps'][idx]
        video_id = eval_df['vid'][idx]
        label = eval_df['text'][idx]
        fts_key = eval_df['fts_key'][idx]
        decoder_input_ids = [prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                         k=int(args.k), is_test=True)]

        # load video
        video_fts = []

        image_fts, selected_frames_indices, full_images_fts = videoUtils.get_clip_video_feats(args.video_dir + file_name)
        input_tensor = torch.tensor(image_fts)

        input_tensor = input_tensor.to(torch.float32)

        # 定义填充量为负无穷，填充到 80x512 FloatTensor
        # padding = torch.tensor(float('-inf'))
        padding = torch.tensor(0)

        output_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, 80 - image_fts.shape[0]), value=padding)
        video_fts.append(output_tensor)

        encoder_outputs = torch.stack(video_fts, dim=0)

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs.to(args.device))

        # with torch.no_grad():
        preds = model.generate(encoder_outputs=encoder_outputs,
                                   decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                                   **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)

        for pred in preds:
            pred = postprocess_preds(pred, tokenizer)
            out.append({"video_id": int(video_id), "file_name": fts_key, "caption": pred, "label": label, "RAG_caps": caps})
            # out_format_str += fts_key + "\t" + pred + "\n"

    return out

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, videoUtils, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    args.txt_file = os.path.join(checkpoint_path, args.outfile_name.rsplit(".", 1)[0] + ".txt")
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, videoUtils, feature_extractor, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)

def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from VideoCaption_Hallucination.model.vision_encoder_decoder import SmallCap, SmallCapConfig
    from VideoCaption_Hallucination.model.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

def main(args):

    videoUtils = VideoUtils()

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.infer_test or args.disable_rag:
    #     args.features_path = None
    #
    # if args.features_path is not None:
    #     feature_extractor = None
    # else:
    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    # if args.disable_rag:
    #     args.k = 0
    #     infer_fn = evaluate_norag_model
    # else:
    #     infer_fn = evaluate_rag_model
    infer_fn = evaluate_model_bz_one

    if args.infer_test:
        # split = 'test'
        split = 'val'
    else:
        split = 'val'

    if args.dataset_type == "MSVD":
        data = videoUtils.load_data_for_inference_MSVD(args.mapping_file, args.annotations_path, args.captions_path)
        eval_df = pd.DataFrame(data['test'])
    if args.dataset_type == "MSR-VTT":
        data = videoUtils.load_data_for_inference_MSR_VTT(args.annotations_path, args.captions_path)
        eval_df = pd.DataFrame(data['test'])



    # args.outfile_name = '{}_{}_preds.json'.format(args.dataset_type, split)

    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH,
                              'no_repeat_ngram_size': 0,
                              'length_penalty': 0.,
                              'num_beams': 1,
                              'early_stopping': True,
                              'eos_token_id': tokenizer.eos_token_id
                              }

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)

        infer_one_checkpoint(args, videoUtils, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, videoUtils, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)



def infer_msvd(checkpoint_path, output_filename, disable_rag=True):
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Model infer')
    # msvd config
    parser.add_argument("--dataset_type", type=str, default="MSVD")
    parser.add_argument("--features_path", type=str,
                        default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_clip_ViT_L_14_fts_four.hdf5",
                        help="Directory where cached input image features are stored")
    parser.add_argument("--sg_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/SG/20/")
    parser.add_argument("--hallucination_model_path", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/VideoCaption_Hallucination/model/dataset_create/msvd_hallucination_identity_001.pt")
    parser.add_argument("--data_root_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/",
                        help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--annotations_path", type=str,
                        default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_caption_format.json",
                        help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--mapping_file", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/youtube_mapping.txt",
                        help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--captions_path", type=str,
                        default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_combine_frame_retrieved_caps_Vit_B32.json",
                        help="JSON file with retrieved captions")
    parser.add_argument("--video_dir", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/video/",
                        help="Directory where input image features are stored")
    parser.add_argument("--checkpoint_path", type=str, default=checkpoint_path,
                        help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")


    parser.add_argument("--experiments_dir", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result/",
                        help="Directory where trained models will be saved")
    parser.add_argument("--model_path", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result", help="Path to model to use for inference")

    parser.add_argument("--infer_test", action="store_true", default=True, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/loc_openai/clip-vit-base-patch32",
                        help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/loc_gpt2",
                        help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=disable_rag,
                        help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64",
                        help="Visual encoder used for retieving captions")
    parser.add_argument("--template_path", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/VideoCaption_Hallucination/model/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")

    args = parser.parse_args()

    args.outfile_name = output_filename
    main(args)

    return FT.time_format(time.time() - start_time)

def infer_msrvtt(checkpoint_path, output_filename, disable_rag=True):
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Model infer')

    # msr-vtt config
    parser.add_argument("--dataset_type", type=str, default="MSR-VTT")
    parser.add_argument("--features_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/msr-vtt_clip_ViT_L_14_fts_four.hdf5")
    parser.add_argument("--data_root_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/")
    parser.add_argument("--hallucination_model_path", type=str,
                        default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/VideoCaption_Hallucination/model/dataset_create/msrvtt_hallucination_identity_001.pt")
    parser.add_argument("--annotations_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json")
    parser.add_argument("--captions_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/msrvtt_combine_frame_retrieved_caps_Vit_B32.json", help="JSON file with retrieved captions")
    parser.add_argument("--video_dir", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/")
    parser.add_argument("--checkpoint_path", type=str, default=checkpoint_path, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")
    parser.add_argument("--sg_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/SG/20/")

    parser.add_argument("--experiments_dir", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result/",
                        help="Directory where trained models will be saved")
    parser.add_argument("--model_path", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/result", help="Path to model to use for inference")

    parser.add_argument("--infer_test", action="store_true", default=True, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/loc_openai/clip-vit-base-patch32",
                        help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/loc_gpt2",
                        help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=disable_rag,
                        help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64",
                        help="Visual encoder used for retieving captions")
    parser.add_argument("--template_path", type=str, default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/VideoCaption_Hallucination/model/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")

    args = parser.parse_args()
    args.outfile_name = output_filename
    main(args)
    return FT.time_format(time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model infer')

    # parser.add_argument("--dataset_name", type=str, default="msr-vtt")
    # parser.add_argument("--dataset_name", type=str, default="msvd")

    # msvd config
    parser.add_argument("--dataset_type", type=str, default="MSVD")
    parser.add_argument("--features_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_clip_ViT_L_14_fts_four.hdf5", help="Directory where cached input image features are stored")
    parser.add_argument("--data_root_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--annotations_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_caption_format.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--mapping_file", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/youtube_mapping.txt", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--captions_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_combine_frame_retrieved_caps_Vit_B32.json", help="JSON file with retrieved captions")
    parser.add_argument("--video_dir", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/video/", help="Directory where input image features are stored")
    parser.add_argument("--checkpoint_path", type=str, default="MSVD_norag_7M_loc_gpt2_40_01_13_22_50/checkpoint-10200", help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")
    parser.add_argument("--sg_path", type=str,default="/home/wy3/zjc_data/datasets/MSVD-QA/SG/20/")

    # parser.add_argument("--checkpoint_path", type=str, default="loc_gpt2/checkpoint-20360", help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    # msr-vtt config
    # parser.add_argument("--dataset_type", type=str, default="MSR-VTT")
    # parser.add_argument("--features_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/msr-vtt_train_clip_fts.hdf5")
    # parser.add_argument("--data_root_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/")
    # parser.add_argument("--annotations_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json")
    # parser.add_argument("--captions_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/msrvtt_mean_retrieved_caps_Vit_B32.json", help="JSON file with retrieved captions")
    # parser.add_argument("--video_dir", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/data/train-video/")
    # parser.add_argument("--checkpoint_path", type=str, default="MSR-VTT_norag_7M_loc_gpt2_40_01_13_22_49/checkpoint-27160", help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")


    parser.add_argument("--experiments_dir", type=str, default="../../result/", help="Directory where trained models will be saved")
    parser.add_argument("--model_path", type=str, default="../../result", help="Path to model to use for inference")


    parser.add_argument("--infer_test", action="store_true", default=True, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="../../loc_openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="../../loc_gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=True, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--template_path", type=str, default="template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size; only matter if evaluating a norag model")

    args = parser.parse_args()

    main(args)
   
