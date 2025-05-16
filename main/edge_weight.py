import jsoni
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from collections import defaultdict
from tqdm import tqdm
import os


# 加载 JSON 文件
def load_json(file_path):
    """加载 JSON 文件并返回其内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_all_distances(json_file_paths):
    all_points = defaultdict(list)

    # 使用 tqdm 显示加载进度
    for file_path in tqdm(json_file_paths, desc="加载JSON文件"):
        data = load_json(file_path)
        for entry in data:
            file_name = entry['file']
            index = entry['index']
            encoding = entry['encoding']
            point = (file_name, index)
            all_points[point].append(encoding)

    # 获取所有点的组合
    points = list(all_points.keys())
    distances = []

    # 使用 tqdm 显示距离计算的进度
    for i in tqdm(range(len(points)), desc="计算距离"):
        for j in range(i + 1, len(points)):
            point_a = points[i]
            point_b = points[j]
            distance = compute_distance(all_points[point_a][0][0], all_points[point_b][0][0])
            distances.append((point_a, point_b, distance))

    return distances


# 计算两个点之间的距离
def compute_distance(encoding_a, encoding_b):
    return euclidean(encoding_a, encoding_b)


# 计算所有点之间的距离
def compute_weighted_cdf(distances):
    distance_values = [dist for (_, _, dist) in distances]
    points_pairs = [(point_a, point_b) for (point_a, point_b, _) in distances]

    # 对距离进行排序，并记录排序后的索引
    sorted_indices = np.argsort(distance_values)
    sorted_distances = np.array(distance_values)[sorted_indices]

    # 计算相邻距离之间的间隔
    intervals = np.diff(sorted_distances, prepend=sorted_distances[0])

    # 计算加权因子 (间隔归一化)
    weights = intervals / np.sum(intervals)

    # 计算加权 CDF
    weighted_cdf = np.cumsum(weights)
    weighted_distances = []
    for (point_a, point_b, dist) in distances:
        # 找到对应的 CDF 值
        index = np.searchsorted(sorted_distances, dist)
        weighted_distances.append((point_a, point_b, weighted_cdf[index]))

    return weighted_distances


# 计算 KDE 密度分布
def grid_density_1d(distances, grid_size):
    distance_values = [dist for (_, _, dist) in distances]
    points_array = np.array(distance_values)
    min_value, max_value = points_array.min(), points_array.max()
    bins = np.linspace(min_value, max_value, grid_size)
    density, _ = np.histogram(points_array, bins=bins)
    max_density_index = np.argmax(density)

    # 获取最大密度区间的中心值
    bin_center = (bins[max_density_index] + bins[max_density_index + 1]) / 2
    return bin_center


# 保存结果到 CSV 文件
def save_distances_to_csv(distances, file_path):
    df = pd.DataFrame(distances, columns=['point_a', 'point_b', 'distance'])
    df.to_csv(file_path, index=False)


# 主函数
def main(parent_folder_path, output_folder_path):
    # 遍历父文件夹中的所有子文件夹
    for subfolder in os.listdir(parent_folder_path):
        subfolder_path = os.path.join(parent_folder_path, subfolder)
        #if subfolder!="_set1":
        #    continue
        # 确保只处理文件夹
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")
            
            # 获取所有JSON文件的路径
            json_file_paths = [os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path) if filename.endswith('.json')]

            # 创建子文件夹输出路径
            sub_output_folder_path = os.path.join(output_folder_path, subfolder)
            os.makedirs(sub_output_folder_path, exist_ok=True)

            for json_file_path in tqdm(json_file_paths, desc=f"处理子文件夹 {subfolder} 的 JSON 文件"):
                # 获取文件名前缀
                prefix = os.path.basename(json_file_path).split('_')[0]
                if "llama" in prefix:
                    continue
                if "qwen" in prefix:
                    continue
                # 在主输出文件夹中为当前子文件夹创建一个独立的子文件夹
                file_specific_output_folder = sub_output_folder_path
                os.makedirs(file_specific_output_folder, exist_ok=True)
                
                # 指定输出的 CSV 文件路径
                output_csv_path = os.path.join(file_specific_output_folder, f"{prefix}_distances_output.csv")
                
                # 计算距离
                distances = calculate_all_distances([json_file_path])
                
                # 计算加权 CDF
                weight_distances = compute_weighted_cdf(distances)
                
                # 计算密度
                density = grid_density_1d(weight_distances, 1000)
                
                # 过滤出密度小于最大密度的距离
                max_density = np.max(density)
                filtered_distances = [(a, b, d) for (a, b, d) in weight_distances if d < max_density]
                
                # 保存到 CSV 文件
                save_distances_to_csv(filtered_distances, output_csv_path)



# 示例调用
if __name__ == "__main__":
    parent_folder_path = "/path"  # 父文件夹路径
    output_folder_path = "/out_path"  # 输出文件夹路径
    main(parent_folder_path, output_folder_path)
