# coding=utf-8
"""
    @project: zero-shot-video-to-text-main
    @Author：no-zjc
    @file： File_Tools.py
    @date：2023/11/14 19:02
"""

import json
import os
import shutil
from datetime import datetime




class File_Tools():

    @staticmethod
    def time_format(time):
        hours, remainder = divmod(time, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = (seconds - int(seconds)) * 1000
        seconds = int(seconds)

        # Return the formatted time string
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d}:{int(milliseconds):03d}"

    @staticmethod
    def get_filenames(path):
        full_paths = []
        file_names = []
        base_names = []

        for root, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                full_paths.append(full_path)

                # 获取文件名（包含后缀）
                file_name = os.path.basename(full_path)
                file_names.append(file_name)

                # 去除文件后缀的文件名
                base_name = os.path.splitext(file_name)[0]
                base_names.append(base_name)

        return full_paths, file_names, base_names

    @staticmethod
    def load_json_data(path):
        with open(path, "r", encoding='utf-8') as f1:
            json_data = json.load(f1)
            f1.close()
        return json_data

    @staticmethod
    def write_to_json(filepath, value):
        with open(filepath, "w", encoding="utf-8") as write_file:
            json.dump(value, write_file, ensure_ascii=False)
            write_file.close()

    @staticmethod
    def write_to_json_timestamp(filepath, value):
        current_time = datetime.now()
        time_stamp = current_time.strftime("%m_%d_%H_%M_%S")
        filepath = filepath.split(".json")[0] + time_stamp + ".json"

        with open(filepath, "w", encoding="utf-8") as write_file:
            json.dump(value, write_file, ensure_ascii=False)
            write_file.close()

    @staticmethod
    def get_timestamp():
        current_time = datetime.now()
        time_stamp = current_time.strftime("%m_%d_%H_%M")

        return time_stamp

    @staticmethod
    def get_subfolders(path):
        subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return subfolders

    @staticmethod
    def get_sub_folder_and_file(directory):
        folders = []
        filenames = []
        # 获取目录下的所有文件和文件夹名
        items = os.listdir(directory)

        # 打印一级文件夹名
        for item in items:
            if os.path.isdir(os.path.join(directory, item)):
                folders.append(item)

        # 打印一级文件名
        for item in items:
            if os.path.isfile(os.path.join(directory, item)):
                filenames.append(item)

        return folders, filenames

    @staticmethod
    def find_files_by_end(directory, tile):
        files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(tile):
                    file_path = os.path.join(root, file)
                    files.append(file_path)
        return files

    @staticmethod
    def copy_files(source_dir, target_dir, tile):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        video_file = os.path.join(target_dir, "video")
        if not os.path.exists(video_file):
            os.makedirs(video_file)
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            if file_name.endswith(tile) and os.path.isfile(source_file):
                target_file = os.path.join(target_dir, file_name)
                shutil.copy2(source_file, target_file)

    @staticmethod
    def write_to_txt(file_path, content):
        with open(file_path, 'w') as file:
            file.write(content)

    @staticmethod
    def file_rename(filepath, new_name):
        # 修改文件或文件夹名字
        os.rename(filepath, os.path.join(os.path.dirname(filepath), new_name))

    @staticmethod
    def find_files(folder_path, extensions):
        '''
        查找指定目录下的所有符合条件的文件路径
        :param folder_path: 目标路径
        :param extensions: 需要查找的文件的后缀
        :return:
        '''
        file_paths = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(extensions):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

        return file_paths

    @staticmethod
    def msr_vtt_split(train_anno_path, save_path, test_anno_path=None):
        """
        0-6512
        6513-7009
        7010-9999
        根据分类的视频文件生成相应的文件
        :param anno_path:
        :return:
        """
        train_anno = File_Tools.load_json_data(train_anno_path)
        train_anno_caption = train_anno.get("sentences")
        msr_vtt_cap = {}
        for ap in train_anno_caption:
            key = ap.get("video_id")
            value = []
            if msr_vtt_cap.get(key) is not None:
                value = msr_vtt_cap.get(key)

            value.append(ap.get("caption"))

            msr_vtt_cap.update({key: value})

        if test_anno_path is not None:
            test_anno = File_Tools.load_json_data(test_anno_path)
            test_anno_caption = test_anno.get("sentences")
            for tp in test_anno_caption:
                key = tp.get("video_id")
                value = []
                if msr_vtt_cap.get(key) is not None:
                    value = msr_vtt_cap.get(key)

                value.append(tp.get("caption"))

                msr_vtt_cap.update({key: value})

        File_Tools.write_to_json(save_path, msr_vtt_cap)

    @staticmethod
    def get_msvd_mapping_dict(mapping_file="/home/wy3/zjc_data/datasets/MSVD-QA/youtube_mapping.txt"):
        # 打开文件
        mapping_dict = {}
        with open(mapping_file, 'r') as file:
            # 逐行读取并按空格拆分后输出
            for line in file.readlines():
                words = line.split()  # 使用 split 方法按空格拆分字符串，默认按空格拆分
                mapping_dict.update({words[0]: words[1]})
        # print(mapping_dict)
        return mapping_dict


    @staticmethod
    def get_count_word_occurrences(word_dict):
        all_word_repeat = []
        for value in word_dict.values():
            all_word_repeat = all_word_repeat + [x for x in value if len(x) > 1]
        word_count = {}
        for word in all_word_repeat:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        sorted_word_occurrences = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_word_occurrences

    @staticmethod
    def get_count_word_occurrences_by_keyword(word_dict, keyword):
        word_repeat = []
        for value in word_dict.values():
            if keyword in value:
                word_repeat = word_repeat + [x for x in value if len(x) > 1]
        word_count = {}
        for word in word_repeat:
            if word in word_count and word != keyword:
                word_count[word] += 1
            else:
                word_count[word] = 1
        sorted_word_occurrences = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_word_occurrences

if __name__ == '__main__':

    # File_Tools.msr_vtt_split("/home/wy3/zjc_data/datasets/MSR-VTT/data/train_val_videodatainfo.json"
    #                                     , "/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json"
    #                                     , "/home/wy3/zjc_data/datasets/MSR-VTT/data/test_videodatainfo.json")
    # data = File_Tools.load_json_data("/home/wy3/zjc_data/datasets/MSR-VTT/data/msrvtt_caption.json")
    # print(data.get("video10001"))

    File_Tools.get_msvd_mapping_dict()
