import pandas as pd, os
from video_noise import NoiseRegistry
import re
from tqdm import tqdm
import sacrebleu
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import trange

rouge_judge = load('rouge')
ROOT_PATH = 'outputs'
noises  = ['impulse', 'motion_blur', 'h265_artifacts'] 
models = [
    # "VideoChat2-HD",
    # "Chat-UniVi-7B-v1.5",
    # "LLaMA-VID-7B",
    # "PLLaVA-13B",
    "Qwen2.5-VL-3B-Instruct",
]
index =  [1, 2, 11, 20, 35, 36, 51, 56, 61, 68, 72, 78, 82, 92, 94, 111, 113, 124, 136, 137, 142, 161, 168, 172, 177, 179, 181, 183, 184, 186, 206, 212, 217, 231, 242, 252, 260, 270, 271, 281, 285, 288, 307, 310, 315, 345, 348, 407, 425, 426, 447, 454, 463, 471, 479, 493, 530, 532, 536, 537, 546, 554, 556, 558, 563, 570, 574, 576, 577, 581, 583, 594, 610, 612, 613, 617, 665, 670, 675, 720, 754, 803, 809, 864, 867, 869, 871, 903, 910, 916, 917, 978, 979, 981, 984, 990, 1011, 1025, 1029, 1057, 1062, 1090, 1093, 1106, 1119, 1120, 1121, 1123, 1124, 1125, 1126, 1128, 1130, 1131, 1133, 1134, 1135, 1143, 1151, 1181, 1185, 1187, 1191, 1197, 1201, 1209, 1214, 1223, 1229, 1257, 1261, 1265, 1275, 1279, 1285, 1326, 1341, 1391, 1420, 1430, 1435, 1451, 1484, 1485, 1486, 1538, 1570, 1588, 1590, 1592, 1617, 1622, 1653, 1682, 1698, 1737, 1740, 1744, 1745, 1747, 1759, 1769, 1802, 1805, 1815, 1835, 1839, 1852, 1856, 1873, 1877, 1886, 1897, 1898, 1911, 1912, 1913, 1918, 1919, 1930, 1931, 1932, 1943, 1944, 1945, 1951, 1952, 1960, 1967, 1968, 1970, 1971, 1975, 1977, 1978, 1979, 1980, 1981]          
pattern = re.compile(r'''
    (?x)                    # 允许空白和注释
    (?:                     # ---- 非捕获组 1: yes/no 后跟 . 或 , ----
        \b                  # 单词边界
        (?:[Yy]es|[Nn]o)    # Yes/yes 或 No/no，后两字母强制小写
        [\.,]               # 紧跟 句号 或 逗号
    )
    |                       # OR
    (?:                     # ---- 非捕获组 2: 整串纯数字（整数或小数） ----
        ^\d+(?:\.\d+)?$     # 例如 123 或 45.67
    )
''', flags=re.VERBOSE)
digital_pattern = re.compile(r'^\d+(?:\.\d+)?$')
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

MMV_DIMENSIONS = {
    'CP': ['Video Topic', 'Video Emotion', 'Video Scene', 'Video Style'],
    'FP-S': ['OCR', 'Object Recognition', 'Attribute Recognition', 'Event Recognition', 'Human Motion', 'Counting'],
    'FP-C': ['Spatial Relationship', 'Human-object Interaction', 'Human Interaction'],
    'HL': ['Hallucination'],
    'LR': ['Structuralized Image-Text Understanding', 'Mathematical Calculation'],
    'AR': ['Physical Property', 'Function Reasoning', 'Identity Reasoning'],
    'RR': ['Natural Relation', 'Physical Relation', 'Social Relation'],
    'CSR': ['Common Sense Reasoning'],
    'TR': ['Counterfactual Reasoning', 'Causal Reasoning', 'Future Prediction'],
}
MMV_DIMENSIONS['Perception'] = []
MMV_DIMENSIONS['Reasoning'] = []
MMV_DIMENSIONS['Overall'] = []
for k in ['CP', 'FP-C', 'FP-S', 'HL']:
    MMV_DIMENSIONS['Perception'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])
for k in ['LR', 'AR', 'RR', 'CSR', 'TR']:
    MMV_DIMENSIONS['Reasoning'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])



def accuracy_score():
    for model in models:
        records = []
        for noise in tqdm(noises):
            coarse_rating = {k: [] for k in MMV_DIMENSIONS}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            
            candidates = [f for f in os.listdir(folder)
                        if f.lower().endswith('0.9.xlsx')][0]
            infile = os.path.join(folder, candidates)

            try:     
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue
            
            for i in index:
                ans_str = str(df.at[i, 'answer'])
                matches = pattern.findall(ans_str)
                cate = str(df.at[i, 'dimensions'])
                cates = eval(cate)

                # 只要正好匹配到一次，就认为是“匹配成功”
                if len(matches) != 1:
                    # 跳过不符合预期的行
                    continue

                matched_text = matches[0]
                pred_str = str(df.at[i, 'prediction'])

                # 判断是数字题还是判断题
                if not digital_pattern.match(matched_text):
                    ans_val = matched_text[:-1].lower()
                    correct = pred_str.lower().startswith(ans_val)
                
                    for d in MMV_DIMENSIONS:
                        if np.any([x in MMV_DIMENSIONS[d] for x in cates]):
                            coarse_rating[d].append(correct)

            coarse_acc = {k: f"{np.mean([max(x, 0) for x in v]):.3f}" for k, v in coarse_rating.items()}
            # 构造表格记录
            if noise == 'origin':
                noise_name = 'clean'
            else:
                noise_name = noise
            # 这里的 noise_type 是为了和原有表格保持一致
            rec = {'noise_type': noise_name}
            rec.update(coarse_acc)
            records.append(rec)

        # 把表格生成 DataFrame 并输出 LaTeX 行
        df = pd.DataFrame(records)
        preprocess(df)
        print(f"\n=== {model} ===")
        print(df)
        cols = df.columns.tolist()
        cols.pop(0)
        for _, row in df.iterrows():
            name = row['noise_type'].replace('_', '\\_')
            vals = ' & '.join(f"{row[c]}" for c in cols)
            print(f"{name} & {vals} \\\\")
            if row['noise_type'] == 'clean':
                print(f"\\hline")
                


def metric_score():
    for model in models:
        records = []
        for noise in tqdm(noises):
            coarse_rating = {k: [] for k in MMV_DIMENSIONS}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            
            candidates = [f for f in os.listdir(folder)
                        if f.lower().endswith('0.9.xlsx')][0]
            infile = os.path.join(folder, candidates)

            try:     
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue
            
            for i in trange(len(df)):
                ans = str(df.at[i, 'answer'])
                predict = str(df.at[i, 'prediction'])

                cate = str(df.at[i, 'dimensions'])
                cates = eval(cate)


                embeddings1 = sbert_model.encode(
                    [predict],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings2 = sbert_model.encode(
                    [ans],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                cosine_scores = util.cos_sim(embeddings1, embeddings2)
                pairwise_scores = cosine_scores.diag().cpu().tolist()[0]
                for d in MMV_DIMENSIONS:
                    if np.any([x in MMV_DIMENSIONS[d] for x in cates]):
                        coarse_rating[d].append(pairwise_scores)

            coarse_acc = {k: f"{round(np.mean([max(x, 0) for x in v]), 4)}" for k, v in coarse_rating.items()}
            # 构造表格记录
            if noise == 'origin':
                noise_name = 'clean'
            else:
                noise_name = noise
            # 这里的 noise_type 是为了和原有表格保持一致
            rec = {'noise_type': noise_name}
            rec.update(coarse_acc)
            records.append(rec)

        # 把表格生成 DataFrame 并输出 LaTeX 行
        df = pd.DataFrame(records)
        preprocess(df)
        print(f"\n=== {model} ===")
        print(df)
        cols = df.columns.tolist()
        cols.pop(0)
        for _, row in df.iterrows():
            name = row['noise_type'].replace('_', '\\_')
            vals = ' & '.join(f"{row[c]}" for c in cols)
            print(f"{name} & {vals} \\\\")
            if row['noise_type'] == 'clean':
                print(f"\\hline")
                



def preprocess(df):
    # 原有列顺序调整逻辑不变
    col_to_move = df.pop('Perception')
    pos  = df.columns.get_loc('HL') + 1
    df.insert(pos, 'Perception', col_to_move)

    col_to_move = df.pop('Reasoning')
    pos  = df.columns.get_loc('TR') + 1
    df.insert(pos, 'Reasoning', col_to_move)


if __name__ == '__main__':
    # accuracy_score()
    metric_score()
    # print(dic)
    # record_to_csv(dic)
    

        