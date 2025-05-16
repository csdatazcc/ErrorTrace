import json
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import os
from tqdm import tqdm

# 指定GPU设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# BERT模型和tokenizer的路径
model_path = '/bert_model'
tokenizer_path = '/bert_tokenizer'

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
bert_model = BertModel.from_pretrained(model_path)
bert_model.to(device)
bert_model.eval()

# 父文件夹路径，包含多个子文件夹
parent_folder_path = '/path'

# 遍历父文件夹中的所有子文件夹
for subfolder in os.listdir(parent_folder_path):
    subfolder_path = os.path.join(parent_folder_path, subfolder)

    # 确保只处理文件夹
    if os.path.isdir(subfolder_path):
        print(f"Processing folder: {subfolder}")

        # 遍历子文件夹中的所有CSV文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(subfolder_path, filename)
                
                # 读取CSV文件，假设sentence列在第一列
                df = pd.read_csv(file_path)

                # 存储每个文件的编码结果
                json_data = []
                threshold = 0.4
                df = df[df["file"]!="imdb"]
                length = len(df[df["weight"]>threshold])
                len_ = len(df[df["weight"]>threshold+0.1])
                while length<2000 or len_<1500:
                    threshold -=0.05
                    length = len(df[df["weight"]>threshold])
                    len_ = len(df[df["weight"]>threshold+0.1])
                df = df[df["weight"]>threshold]
                # 使用 tqdm 显示进度条
                for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Processing {filename}', unit='sentence'):
                    # 获取sentence列内容（根据实际情况调整列名）
                    sentence = row['sentence']
                    file = row['file']
                    index = row['index']

                    # 使用BERT tokenizer对句子进行编码
                    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
                    inputs = {key: value.to(device) for key, value in inputs.items()}

                    # 获取BERT模型的输出
                    with torch.no_grad():
                        outputs = bert_model(**inputs)

                    # 获取[CLS] token的输出作为句子的表示
                    semantic_encoding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

                    # 构建JSON结构
                    json_entry = {
                        "file": file,
                        "index": index,
                        "encoding": semantic_encoding
                    }
                    json_data.append(json_entry)

                # 保存为JSON文件
                output_file = f"{os.path.splitext(filename)[0]}_semantic_encodings.json"
                output_path = os.path.join(subfolder_path, output_file)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)

                print(f"Semantic encodings saved to {output_path}")
