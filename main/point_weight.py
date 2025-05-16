import math
import os
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


# 从 CSV 文件中加载数据
def load_csv(file_path):
    return pd.read_csv(file_path)

model_name_ = "llama70b"
model_name_0 = "llama3_70b"
# 根据文件名确定族群
def determine_group(file_name):
    if re.search(r'tral', file_name, re.IGNORECASE):
        return 'mistral'
    elif re.search(r'llama', file_name, re.IGNORECASE):
        return 'llama'
    elif re.search(r'qwen', file_name, re.IGNORECASE):
        return 'qwen'
    elif re.search(r'gemma', file_name, re.IGNORECASE):
        return 'gemma'
    else:
        return 'unknown'


# 将数据按照族群组织
def organize_by_file(csv_file_paths):
    group_dict = defaultdict(list)
    file_count_dict = defaultdict(list)  # 统计每个族群的文件数量
    for file_path in csv_file_paths:
        file_name = os.path.basename(file_path)
        result = re.search(r'^(.*?)(_error)', file_name)
        model_name = result.group(1)
        #if model_name == model_name_:
         #   print(model_name)
         #   continue
        #if model_name == model_name_0:
        #    print(model_name)
        #    continue
        '''
        if model_name == "qwen2_7b":
            print(model_name)
            continue
        '''
        group = determine_group(model_name)
        df = load_csv(file_path)
        file_count_dict[group].append(model_name)  # 记录每个族群的文件
        # 将 DataFrame 转换为包含 'file' 和 'index' 字段的字典
        for _, row in df.iterrows():

            group_dict[group].append({
                'model': model_name,
                'file': row['file'],
                'index': row['index'],
                'sentence': row['sentence']  # 提取句子
            })
    # 将文件集合的长度（即文件数量）替换集合本身
    print(file_count_dict)
    return group_dict, file_count_dict


# 提取所有唯一的点
def extract_points(group_entries):
    points = set()
    for entry in group_entries:
        point = (entry['file'], entry['index'])  # 通过 file 和 index 识别点
        points.add(point)
    return points


# 计算频率
def calculate_frequencies(point, group, group_entries, other_group_entries, file_count_dict):
    # 计算当前族群中包含该点的次数
    file = point[0]
    in_group_model = [entry['model'] for entry in group_entries if (entry['file'], entry['index']) == point]
    #print(f"in_group_model:{in_group_model}")
    in_group_total = file_count_dict[group]
    in_group_model_value = 0
    in_group_total_value = 0
    for model in in_group_model:
        values = error_rate_data[error_rate_data['file'] == file][model].values
        if values.size == 1:  # 确保数组大小为 1
            in_group_model_value += values.item()
        elif values.size == 0:
            print(f"Warning: No data found for model {model} and file {file}.")
        else:
            print(f"Warning: Multiple values found for model {model} and file {file}. Using the first value.")
            in_group_model_value += values[0]  # 使用第一个值
    for model in in_group_total:
        values = error_rate_data[error_rate_data['file'] == file][model].values
        if values.size == 1:  # 确保数组大小为 1
            in_group_total_value += values.item()
        elif values.size == 0:
            print(f"Warning: No data found for model {model} and file {file}.")
        else:
            print(f"Warning: Multiple values found for model {model} and file {file}. Using the first value.")
            in_group_model_value += values[0]  # 使用第一个值
    # 计算其他族群中包含该点的次数
    out_weight = 0
    total_out_group_files = 0
    for k, v in other_group_entries.items():
        out_group_model = [entry['model'] for entry in v if (entry['file'], entry['index']) == point]
        out_group_total = file_count_dict[k]
        out_group_model_value = 0
        out_group_total_value = 0
        for model in out_group_model:
            values = error_rate_data[error_rate_data['file'] == file][model].values
            if values.size == 1:  # 确保数组大小为 1
                out_group_model_value += values.item()
            elif values.size == 0:
                print(f"Warning: No data found for model {model} and file {file}.")
            else:
                print(f"Warning: Multiple values found for model {model} and file {file}. Using the first value.")
                out_group_model_value += values[0]  # 使用第一个值
        for model in out_group_total:
            values = error_rate_data[error_rate_data['file'] == file][model].values
            if values.size == 1:  # 确保数组大小为 1
                out_group_total_value += values.item()
            elif values.size == 0:
                print(f"Warning: No data found for model {model} and file {file}.")
            else:
                print(f"Warning: Multiple values found for model {model} and file {file}. Using the first value.")
                out_group_total_value += values[0]  # 使用第一个值
        weight = (out_group_model_value/out_group_total_value)
        out_weight += weight ** 2
        total_out_group_files += weight
    #print(f'in_group_model_value:{in_group_model_value}')
    #print(f'in_group_total_value:{in_group_total_value}')
    #print(f'out_weight:{out_weight}')
    #print(f'total_out_group_files:{total_out_group_files}')
    in_group_freq = in_group_model_value / in_group_total_value if in_group_total_value > 0 else 0
    # 计算其他族群中的频率
    out_group_freq = out_weight / total_out_group_files if total_out_group_files > 0 else 0

    return in_group_freq, out_group_freq


# 计算权重
def calculate_weight(in_group_freq, out_group_freq):
    return in_group_freq * (1 - out_group_freq)


# 将权重和句子保存为 CSV 文件
def save_weights_to_csv(weights, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在`
    for group, points in weights.items():
        # 为每个族群创建一个 DataFrame
        data = []
        for (file, index), (weight, sentence) in points.items():
            data.append({'file': file, 'index': index, 'weight': weight, 'sentence': sentence})
        df = pd.DataFrame(data)

        # 将每个族群的数据保存到指定输出文件夹中的 CSV 文件
        output_file_path = os.path.join(output_dir, f"{group}.csv")
        df.to_csv(output_file_path, index=False)


# 从指定文件夹中获取所有 CSV 文件
def get_csv_files_from_folder(folder_path):
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def main(csv_folder_path, output_dir):
    # 获取文件夹中的所有 CSV 文件
    csv_file_paths = get_csv_files_from_folder(csv_folder_path)
    # 加载所有 CSV 文件并组织数据
    groups, file_count_dict = organize_by_file(csv_file_paths)

    # 仅保留 `qwen` 族群的数据

    # 存储每个族群中每个点的权重和句子
    weights = defaultdict(dict)

    for group, group_entries in groups.items():
        other_group_entries = {k: v for k, v in groups.items() if k != group}
        # 提取当前 group 中唯一的点集
        group_points = extract_points(group_entries)

        # 使用 tqdm 显示进度条
        for point in tqdm(group_points, desc=f'Processing {group}', unit='points'):
            file_names = point[0]
            if file_names =="imdb":
                continue 
            in_group_freq, out_group_freq = calculate_frequencies(
                point, group, group_entries, other_group_entries, file_count_dict)

            if out_group_freq >= in_group_freq:
                continue

            weight = calculate_weight(in_group_freq, out_group_freq)
            # 仅在权重大于等于0.5时添加到权重字典
            if weight >= 0.05:
                # 找到相应的句子
                sentence = next(
                    (entry['sentence'] for entry in group_entries if (entry['file'], entry['index']) == point), '')
                weights[group][point] = (weight, sentence)
    save_weights_to_csv(weights, output_dir)

if __name__ == "__main__":
    csv_folder_path = "csv_/path"  # 替换为你的CSV文件夹路径
    output_dir = "/out_path"  # 输出 CSV 文件的路径（会为每个族群生成单独的文件）
    error_rate = "/rate_path"
    error_rate_data = load_csv(error_rate)
    main(csv_folder_path, output_dir)
