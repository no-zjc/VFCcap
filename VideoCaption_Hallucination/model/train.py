import pandas as pd
import numpy as np
import os
import argparse

from VideoCaption_Hallucination.video_utils import VideoUtils, TrainDataset

os.environ["WANDB_DISABLED"] = "true"

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM, IntervalStrategy
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from transformers import VisionEncoderDecoderModel, CLIPModel, CLIPVisionModel, EncoderDecoderModel
from VideoCaption_Hallucination.model.vision_encoder_decoder import SmallCap, SmallCapConfig
from VideoCaption_Hallucination.model.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from VideoCaption_Hallucination.File_Tools import File_Tools as FT

# for attention with 28M params, we devide the attention dimensions by 1
# for attention with 14M params, we devide the attention dimensions by 2, etc.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25


def get_model_and_auxiliaries(args):
    # register model types

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    model = SmallCap.from_encoder_decoder_pretrained(args.encoder_name, args.decoder_name,
                                                     cross_attention_reduce_factor=cross_attention_reduce_factor)
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder
    model.config.max_length = CAPTION_LENGTH
    model.config.rag = not args.disable_rag

    # print("model",model)
    # print(stop)
    # freeze parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    if "xglm" in args.decoder_name or "opt" in args.decoder_name:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'encoder_attn' not in name:
                    param.requires_grad = False

    else:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False

    # count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {} trainable parameters.'.format(num_trainable_params))

    return model, tokenizer, feature_extractor


def get_data(tokenizer, max_length, args):
    videoUtils = VideoUtils()
    if args.dataset_type == 'MSVD':
        data = videoUtils.load_data_for_training_MSVD(args.mapping_file, args.annotations_path,  args.captions_path)
    elif args.dataset_type == 'MSR-VTT':
        data = videoUtils.load_data_for_training_MSR_VTT(args.annotations_path, args.captions_path)
    train_df = pd.DataFrame(data['train'])

    if args.ablation_visual:
        print("ablation_visual is True")
        # train_dataset = AblationFeaturesDataset(
        #                     df=train_df,
        #                     features_path=os.path.join(args.features_dir, 'train.hdf5'),
        #                     tokenizer=tokenizer,
        #                     rag=not args.disable_rag,
        #                     template_path=args.template_path,
        #                     k=args.k,
        #                     max_caption_length=max_length)
    else:
        train_dataset = TrainDataset(
            df=train_df,
            features_path=args.features_path,
            tokenizer=tokenizer,
            rag=not args.disable_rag,
            template_path=args.template_path,
            k=args.k,
            max_caption_length=max_length)

    return train_dataset


def main(args):
    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)
    print(train_dataset.__len__())
    time_str = FT.get_timestamp()

    model_type = 'norag' if args.disable_rag else 'rag'
    if args.ablation_visual:
        output_dir = '{}_{}M_{}_ablation'.format(model_type, args.attention_size, args.decoder_name)
    else:
        output_dir = '{}_{}_{}M_{}_{}_t{}_bs{}'.format(args.dataset_type, model_type, args.attention_size, args.decoder_name.split("/")[-1], args.n_epochs, time_str, args.batch_size)

    output_dir = os.path.join(args.experiments_dir, output_dir)

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs,
        logging_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )

    trainer.train()

def train_msrvtt():
    parser = argparse.ArgumentParser(description='Model Training')

    # msr-vtt config
    parser.add_argument("--dataset_type", type=str, default="MSR-VTT")
    parser.add_argument("--features_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/msr-vtt_clip_ViT_L_14_fts_four.hdf5")
    parser.add_argument("--data_root_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/")
    parser.add_argument("--annotations_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json")
    parser.add_argument("--captions_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/msrvtt_combine_frame_retrieved_caps_Vit_B32.json", help="JSON file with retrieved captions")
    parser.add_argument("--sg_path", type=str, default="/home/wy3/zjc_data/datasets/MSR-VTT/SG/20/")
    parser.add_argument("--hallucination_model_path", type=str,
                        default="")
    parser.add_argument("--experiments_dir", type=str, default="../../result/",
                        help="Directory where trained models will be saved")
    parser.add_argument("--encoder_name", type=str, default="../../loc_openai/clip-vit-base-patch32",
                        help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="../../loc_gpt2",
                        help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--attention_size", type=float, default=7,
                        help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False)

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="Vit-B/32",
                        help="Visual encoder used for retieving captions")
    parser.add_argument("--template_path", type=str, default="template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--ablation_visual", action="store_true", default=False,
                        help="Whether to blank visual features")

    args = parser.parse_args()

    main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')

    # msvd config
    parser.add_argument("--dataset_type", type=str, default="MSVD")
    parser.add_argument("--features_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_clip_ViT_L_14_fts_four.hdf5", help="Directory where cached input image features are stored")
    parser.add_argument("--data_root_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--annotations_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_video_caption_format.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--mapping_file", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/youtube_mapping.txt", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--captions_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/msvd_combine_frame_retrieved_caps_Vit_B32.json", help="JSON file with retrieved captions")
    parser.add_argument("--sg_path", type=str, default="/home/wy3/zjc_data/datasets/MSVD-QA/SG/20/")
    parser.add_argument("--hallucination_model_path", type=str,
                        default="/home/wy3/zjc_data/PyProject/zero-shot-video-to-text/VideoCaption_Hallucination/model/dataset_create/msvd_hallucination_identity_003.pt")

    parser.add_argument("--experiments_dir", type=str, default="../../result/", help="Directory where trained models will be saved")
    parser.add_argument("--encoder_name", type=str, default="../../loc_openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="../../loc_gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--attention_size", type=float, default=7, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False)

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="Vit-L/14", help="Visual encoder used for retieving captions")
    parser.add_argument("--template_path", type=str, default="template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--ablation_visual", action="store_true", default=False,
                        help="Whether to blank visual features")

    args = parser.parse_args()

    main(args)
