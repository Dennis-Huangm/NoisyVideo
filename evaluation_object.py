import pandas as pd
import os
import re
from tqdm import tqdm
from video_noise import NoiseRegistry
import sacrebleu
from evaluate import load
from sentence_transformers import SentenceTransformer, util

# Configuration
ROOT_PATH = 'outputs'
noises = ['origin'] + NoiseRegistry.list_noises()
models = [
    # "VideoChat2-HD",
    # "Chat-UniVi-7B-v1.5",
    # "LLaMA-VID-7B",
    # "PLLaVA-13B",
    "Qwen2.5-VL-3B-Instruct",
]
index =  [1, 2, 11, 20, 35, 36, 51, 56, 61, 68, 72, 78, 82, 92, 94, 111, 113, 124, 136, 137, 142, 161, 168, 172, 177, 179, 181, 183, 184, 186, 206, 212, 217, 231, 242, 252, 260, 270, 271, 281, 285, 288, 307, 310, 315, 345, 348, 407, 425, 426, 447, 454, 463, 471, 479, 493, 530, 532, 536, 537, 546, 554, 556, 558, 563, 570, 574, 576, 577, 581, 583, 594, 610, 612, 613, 617, 665, 670, 675, 720, 754, 803, 809, 864, 867, 869, 871, 903, 910, 916, 917, 978, 979, 981, 984, 990, 1011, 1025, 1029, 1057, 1062, 1090, 1093, 1106, 1119, 1120, 1121, 1123, 1124, 1125, 1126, 1128, 1130, 1131, 1133, 1134, 1135, 1143, 1151, 1181, 1185, 1187, 1191, 1197, 1201, 1209, 1214, 1223, 1229, 1257, 1261, 1265, 1275, 1279, 1285, 1326, 1341, 1391, 1420, 1430, 1435, 1451, 1484, 1485, 1486, 1538, 1570, 1588, 1590, 1592, 1617, 1622, 1653, 1682, 1698, 1737, 1740, 1744, 1745, 1747, 1759, 1769, 1802, 1805, 1815, 1835, 1839, 1852, 1856, 1873, 1877, 1886, 1897, 1898, 1911, 1912, 1913, 1918, 1919, 1930, 1931, 1932, 1943, 1944, 1945, 1951, 1952, 1960, 1967, 1968, 1970, 1971, 1975, 1977, 1978, 1979, 1980, 1981]          
# Patterns for matching answers
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

# Models for advanced metrics
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
rouge_judge = load('rouge')


def load_data():
    """
    Load all Excel files into a single DataFrame, annotated by model, noise, and video_type.
    Expects a 'video_type' column in each sheet.
    """
    frames = []
    for model in models:
        for noise in noises:
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            files = [f for f in os.listdir(folder) if f.lower().endswith('0.9.xlsx')]
            print(files)
            if not files:
                continue
            df = pd.read_excel(os.path.join(folder, files[0]))
            df['model'] = model
            df['noise'] = noise
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score_metrics(all_df):
    """
    Compute accuracy, judge/digital accuracies, BLEU, and BERT for each (model, noise, video_type).
    Returns a DataFrame with those metrics.
    """
    records = []
    groups = all_df.groupby(['model', 'noise', 'video_type'])
    for (model, noise, vtype), group in tqdm(groups, total=len(groups)):
        total = 0
        acc_num = acc_judge = acc_digital = 0
        num_judge = num_digital = 0
        bleu_scores = []
        preds, refs = [], []

        # Accuracy
        for i in index:
            mask = group['index'] == i
            if not mask.any():
                continue

            total += 1
            # 取出 answer 列中对应原始行号 i 的那一行
            ans_str = str(group.loc[mask, 'answer'].iloc[0])
            matches = pattern.findall(ans_str)

            # 只要正好匹配到一次，就认为是“匹配成功”
            if len(matches) != 1:
                # 跳过不符合预期的行
                continue

            matched_text = matches[0]
            pred_str = str(group.loc[mask, 'prediction'].iloc[0])

            # 判断是数字题还是判断题
            if digital_pattern.match(matched_text):
                num_digital += 1
                if matched_text in pred_str:
                    acc_digital += 1; acc_num += 1
            else:
                num_judge += 1
                if pred_str.lower().startswith(matched_text[:-1].lower()):
                    acc_judge += 1; acc_num += 1


        for _, row in group.iterrows():
            ans = str(row['answer'])
            pred = str(row['prediction'])
            # BLEU
            bleu_scores.append(sacrebleu.sentence_bleu(pred, [ans], smooth_method='exp').score)
            preds.append(pred); refs.append(ans)
            
        acc = round(acc_num / total, 3) if total else None
        acc_j = round(acc_judge / num_judge, 3) if num_judge else None
        acc_d = round(acc_digital / num_digital, 3) if num_digital else None
        bleu = round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else None

        # BERT similarity
        emb1 = sbert_model.encode(preds, batch_size=32, convert_to_tensor=True)
        emb2 = sbert_model.encode(refs, batch_size=32, convert_to_tensor=True)
        cos_scores = util.cos_sim(emb1, emb2).diag().cpu().tolist()
        bert_score = round(sum(cos_scores) / len(cos_scores), 4) if cos_scores else None

        records.append({
            'model': model,
            'noise': noise,
            'video_type': vtype,
            'acc': acc,
            'acc_judge': acc_j,
            'acc_digital': acc_d,
            'bleu': bleu,
            'bert': bert_score,
        })

    return pd.DataFrame(records)


def print_latex_tables(metrics_df):
    """
    Print only the LaTeX row entries for each model and metric, in noise_type x video_type order.
    """
    metrics = ['acc', 'acc_judge', 'acc_digital', 'bleu', 'bert']
    for model in sorted(metrics_df['model'].unique()):
        df_m = metrics_df[metrics_df['model'] == model]
        for metric in metrics:
            print(f"\n=== {model} - {metric} ===")
            # pivot to noise vs video_type
            pivot = df_m.pivot(index='noise', columns='video_type', values=metric).reset_index()
            pivot = pivot.rename(columns={'noise': 'noise_type'})
            cols = pivot.columns.tolist()
            cols.pop(0)  # remove 'noise_type'
            # print rows as LaTeX
            for _, row in pivot.iterrows():
                name = row['noise_type'].replace('_', '\\_')
                vals = ' & '.join(f"{row[c]:.3f}" if isinstance(row[c], float) else f"{row[c]}" for c in cols)
                print(f"{name} & {vals} \\\\")
            print()



if __name__ == '__main__':
    all_df = load_data()
    metrics_df = score_metrics(all_df)
    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv('metrics.csv', index=False)
    print_latex_tables(metrics_df)
