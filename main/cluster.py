import csv
from collections import deque
import os
from tqdm import tqdm  # 导入 tqdm 用于进度条显示
import pandas as pd
# 读取点数据文件
def load_points(filename, min_threshold=0.4, max_points=2000, secondary_max_points=1500, step=0.05):

    # 加载点数据为 DataFrame
    df = pd.read_csv(filename)

    # 筛除特定文件的行
    df = df[df["file"] != "imdb"]

    # 动态调整阈值
    threshold = min_threshold
    length = len(df[df["weight"] > threshold])
    len_ = len(df[df["weight"] > threshold + 0.1])

    while length < max_points or len_ < secondary_max_points:
        threshold -= step
        length = len(df[df["weight"] > threshold])
        len_ = len(df[df["weight"] > threshold + 0.1])

    # 筛选权重符合阈值的点
    filtered_df = df[df["weight"] > threshold]

    # 转换为字典 {point_id: weight}
    points = {
        (row["file"], int(row["index"])): row["weight"]
        for _, row in filtered_df.iterrows()
    }
    return points

# 读取边数据文件
def load_edges(filename):
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        edges = {}
        for row in reader:
            point_a = eval(row['point_a'])
            point_b = eval(row['point_b'])
            if point_a not in edges:
                edges[point_a] = {}
            if point_b not in edges:
                edges[point_b] = {}
            distance = float(row['distance'])
            edges[point_b][point_a] = distance
            edges[point_a][point_b] = distance
    return edges

# 寻找密集集合的算法
def find_dense_sets(points, edges):
    C = set()
    W = set(points.keys())
    total_points = len(points)
    dense_sets_info = []
    i = 0
    k = 0
    with tqdm(total=len(W), desc="Finding dense sets", unit="point") as progress:
        while W:
            p = max(W, key=lambda x: points[x])
            i += 1
            Q = deque([p])
            T = set([p])
            point_weight_sum = points[p]
            edge_weight_sum = 0
            k += 1
            W.remove(p)
            while Q:
                x = Q.popleft()
                if x not in edges:
                    continue
                for v, edge_weight in list(edges[x].items()):
                    if v in W:
                        benefits = {}
                        benefits[x] = points[x] - edge_weight
                        if v not in edges:
                            continue
                        for u, edge_weight_b in edges[v].items():
                            if u in W:
                                benefit = points[u] - edge_weight_b
                                benefits[u] = benefit
                        sorted_benefit = dict(sorted(benefits.items(), key=lambda x: x[1], reverse=True))
                        total_items = len(sorted_benefit)
                        judge_benefit = list(sorted_benefit.keys())[:max(int(total_items * 0.5), 1)]
                        if x in judge_benefit and points[v] - edge_weight > 0:
                            Q.append(v)
                            T.add(v)
                            W.remove(v)
                            point_weight_sum += points[v]
                            edge_weight_sum += edge_weight
                            i += 1
                            del edges[x][v]
                progress.update(1)
            remaining_weight = point_weight_sum - edge_weight_sum
            importance = remaining_weight + (len(T) / total_points)
            dense_sets_info.append((T, remaining_weight, importance))
    return dense_sets_info

# 保存结果到文件
def save_important_set(filename, important_set, points, remaining_weight, importance):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File', 'Index', "Weight", 'Remaining Weight', 'Importance'])
        for point in important_set:
            file_name, index = point
            writer.writerow([file_name, index, points[point], remaining_weight, importance])

# 主程序
def process_folders(point_folder, edge_folder, output_folder):
    # 获取 `point_folder` 和 `edge_folder` 中的子文件夹列表
    point_subfolders = {folder: os.path.join(point_folder, folder) for folder in os.listdir(point_folder) if os.path.isdir(os.path.join(point_folder, folder))}
    edge_subfolders = {folder: os.path.join(edge_folder, folder) for folder in os.listdir(edge_folder) if os.path.isdir(os.path.join(edge_folder, folder))}
    
    # 取交集，确保子文件夹名称匹配
    common_subfolders = set(point_subfolders.keys()).intersection(edge_subfolders.keys())

    for subfolder in tqdm(common_subfolders, desc="Processing folders", unit="folder"):
        point_csv_files = [os.path.join(point_subfolders[subfolder], f) for f in os.listdir(point_subfolders[subfolder]) if f.endswith('.csv')]
        edge_csv_files = [os.path.join(edge_subfolders[subfolder], f) for f in os.listdir(edge_subfolders[subfolder]) if f.endswith('.csv')]

        # 遍历点文件和边文件，进行匹配
        for point_file in tqdm(point_csv_files, desc=f"Processing points in {subfolder}", unit="file", leave=False):
            point_prefix = os.path.splitext(os.path.basename(point_file))[0]  # 点文件的前缀
            # 匹配边文件
            matching_edge_files = [f for f in edge_csv_files if point_prefix in os.path.splitext(os.path.basename(f))[0]]

            for edge_file in tqdm(matching_edge_files, desc=f"Processing edges in {subfolder}", unit="file", leave=False):
                # 加载点和边数据
                points = load_points(point_file)
                edges = load_edges(edge_file)

                # 运行算法
                dense_sets_info = find_dense_sets(points, edges)

                # 找到最重要的集合
                most_important_set = max(dense_sets_info, key=lambda x: x[2])

                # 创建输出文件夹
                output_subfolder_path = os.path.join(output_folder, subfolder)
                os.makedirs(output_subfolder_path, exist_ok=True)

                # 保存结果
                output_file = os.path.join(output_subfolder_path, f"{point_prefix}_important_set.csv")
                save_important_set(output_file, most_important_set[0], points, most_important_set[1], most_important_set[2])

# 示例调用
if __name__ == '__main__':
    point_folder = "/point_path"  # 点文件主文件夹路径
    edge_folder = "/edge_path"  # 边文件主文件夹路径
    output_folder = "out_path"  # 输出文件夹路径
    process_folders(point_folder, edge_folder, output_folder)
