import pandas as pd
import os
directory_path = "/error_space_path"
source_dir= "acc_rate_csv"
error_data= "/error_path"
df = pd.read_csv(source_dir)
data = {}
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        df_set = pd.read_csv(file_path)
        file_name = filename.split(".csv")[0] 
        df_set = df_set[df_set["file"] != "imdb"]
        file_ratio = df_set['file'].value_counts(normalize=True)
        data[file_name] = file_ratio
error_rate = {}
difference = {
        "gemma":0,
        "qwen":0,
        "llama":0,
        "mistral":0
}
for _,row in df.iterrows():
    if row[0] == "Over":
        continue
    if row[0] =="IMDB":
        continue
    if row[0] == "commonsense_qa":
        row[0] = "qa"
    error_rate[row[0]] = 1-float(row[1])
print(error_rate)
group_set = ["mistral","gemma","qwen","llama"]
df_set = pd.read_csv(error_data)
#directory_path = '/home/chuanchao/Error_space/importants'
average_increases = {}
suspicious_model = []
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        df_set_now = df_set[df_set["Group"]==filename]
        df_set_now = df_set_now[df_set_now["file"] != "imdb"]
        file_name = filename.split(".csv")[0] 
        df = pd.read_csv(file_path)
        file_ratio = df_set_now['file'].value_counts(normalize=True)
        error = 0
        for k, v in error_rate.items():
            error += file_ratio.get(k, 0) * v
        file_name = filename.split(".csv")[0] 
        error_sentence = len(df_set_now) / len(df)
        print(f"{filename}:{error_sentence}")
        if error_sentence > -0.6:
            suspicious_model.append(filename)
for filename in suspicious_model:
    threshold = 1
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        previous_error_sentence = None  # 用于存储上一个 error_sentence
        increases = []  # 用于存储涨幅
        df_set_now = df_set[df_set["Group"]==filename]
        df = pd.read_csv(file_path)
        length = len(df[df["weight"]>threshold])
        len_ = len(df[df["weight"]>threshold+0.1])
        dis = 600
        while length<1200 or len_<dis:
            threshold  -= 0.05
            length = len(df[df["weight"]>threshold])
            len_ = len(df[df["weight"]>threshold+0.1])
        print(f"{filename}:{threshold},{length}")
        if "mistral" in filename:
            threshold = 0.249999
        ka = 1
        while threshold < 1:
            ka+=1
            if ka > 4:
                break
            df = df[df["weight"]>threshold]
            length = len(df)
            if length<dis:
                break
            df_set_now = df_set[df_set["Group"]==filename]
            df_set_now = df_set_now[df_set_now["weight"]>threshold]
            file_ratio = df_set_now['file'].value_counts(normalize=True)
            error = 0
            for k, v in error_rate.items():
                error += file_ratio.get(k, 0) * v
            file_name = filename.split(".csv")[0] 
            error_sentence = df_set_now["weight"].sum() / df["weight"].sum() - error
            print(f"{filename}:{error_sentence}")
            if previous_error_sentence is not None:
                increase = error_sentence - previous_error_sentence
                increases.append(increase)  # 记录涨幅
                file_index = filename.find(".csv")
                _file = filename[:file_index]
                if increase >0:
                    difference[_file] +=1
                elif increase <0:
                    difference[_file] -=1
            previous_error_sentence = error_sentence
            threshold += 0.05
            
        # 计算该文件的平均涨幅
        if increases:
            average_increase = sum(increases) / len(increases)
            average_increases[filename] = average_increase  # 存储文件的平均涨幅

# 找到平均涨幅最大的文件及其平均涨幅
print(average_increases)
aaaa = []
for ai in average_increases.keys():
    aaaa.append(float(average_increases[ai]))
print(aaaa)
print(difference)
if average_increases:
    max_increase_file = max(average_increases, key=average_increases.get)
    max_increase_value = average_increases[max_increase_file]
    print(f"文件 {max_increase_file} 的平均涨幅最大，平均涨幅为: {max_increase_value}")
else:
    print("没有计算到有效的平均涨幅。")i
