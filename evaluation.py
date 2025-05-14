import pandas as pd
import os
import re
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer, util
import numpy as np
import argparse
import json

judge_index = [   2,   56,   78,   82,   94,  111,  113,  124,  136,  137,  142,  161,  168,  172,  177,  179,
                181,  183,  184,  186,  206,  212,  217,  231,  242,  252,  260,  270,  271,  281,  285,  288,
                307,  310,  315,  345,  348,  407,  426,  447,  454,  463,  471,  479,  493,  530,  532,  536,
                537,  546,  554,  556,  558,  563,  576,  583,  610,  612,  613,  665,  670,  675,  720,  754,
                803,  809,  864,  910,  916,  917, 1025, 1029, 1057, 1062, 1093, 1119, 1120, 1121, 1124, 1125,
               1126, 1130, 1131, 1133, 1134, 1135, 1143, 1151, 1181, 1185, 1214, 1265, 1275, 1341, 1391, 1430,
               1435, 1451, 1484, 1485, 1486, 1588, 1590, 1592, 1617, 1622, 1653, 1682, 1698, 1737, 1740, 1744,
               1745, 1747, 1759, 1769, 1802, 1805, 1815, 1835, 1839, 1852, 1856, 1873, 1877, 1886, 1897, 1898,
               1911, 1912, 1913, 1918, 1919, 1930, 1931, 1932, 1943, 1944, 1945, 1951, 1952, 1960, 1967, 1968,
               1970, 1971, 1975, 1977, 1978, 1979, 1980, 1981 ]

pattern = re.compile(r'''
    (?x)                    # 允许空白和注释
    (?:                     # ---- 非捕获组 1: yes/no 后跟 . 或 , ----
        \b                  # 单词边界
        (?:[Yy]es|[Nn]o)    # Yes/yes 或 No/no
        [\.,]               # 紧跟 句号 或 逗号
    )
    |                       # OR
    (?:                     # ---- 非捕获组 2: 整串纯数字（整数或小数） ----
        ^\d+(?:\.\d+)?$  # 例如 123 或 45.67
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
L3_DIMS = []
for k, v in MMV_DIMENSIONS.items():
    L3_DIMS.extend(v)

MMV_DIMENSIONS['Perception'] = []
MMV_DIMENSIONS['Reasoning'] = []
MMV_DIMENSIONS['Overall'] = []
for k in ['CP', 'FP-C', 'FP-S', 'HL']:
    MMV_DIMENSIONS['Perception'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])
for k in ['LR', 'AR', 'RR', 'CSR', 'TR']:
    MMV_DIMENSIONS['Reasoning'].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS['Overall'].extend(MMV_DIMENSIONS[k])


def accuracy_score_qtype():
    for model in models:
        for noise in tqdm(noises, desc=f"Model={model} processing acc"):
            coarse_rating = {k: [] for k in MMV_DIMENSIONS}
            fine_rating = {k: [] for k in L3_DIMS}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            candidates = [f for f in os.listdir(folder) if f.lower().endswith(f'{ratio}.xlsx')][0]
            infile = os.path.join(folder, candidates)
            try:
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue
            for i in judge_index:
                ans_str = str(df.at[i, 'answer'])
                matches = pattern.findall(ans_str)
                if len(matches) != 1:
                    continue
                pred_str = str(df.at[i, 'prediction'])
                cates = eval(str(df.at[i, 'dimensions']))

                matched_text = matches[0]
                ans_val = matched_text[:-1].lower()
                correct = pred_str.lower().startswith(ans_val)

                for c in cates:
                    fine_rating[c].append(correct)

                for d in MMV_DIMENSIONS:
                    if any(x in MMV_DIMENSIONS[d] for x in cates):
                        coarse_rating[d].append(correct)

            coarse_acc = {k: f"{np.mean([int(x) for x in v]):.3f}" for k, v in coarse_rating.items()}
            fine_acc = {k: f'{round(np.mean([max(x, 0) for x in v]), 4)}' for k, v in fine_rating.items()}
            rec = dict(coarse_acc=coarse_acc, fine_acc=fine_acc)

            pth = os.path.join(folder, f"{model}_{noise}_{ratio}_acc_qtype.json")
            print(f"\n=== {model} ACC ===")
            print(json.dumps(rec, indent=4))
            json.dump(rec, open(pth, 'w'), indent=4, ensure_ascii=False)


def accuracy_score_vtype():
    for model in models:
        for noise in tqdm(noises, desc=f"Model={model} processing acc"):
            Video_Type = {}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            candidates = [f for f in os.listdir(folder) if f.lower().endswith(f'{ratio}.xlsx')][0]
            infile = os.path.join(folder, candidates)
            try:
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue

            groups = df.groupby(['video_type'])
            for (vtype,), group in tqdm(groups, total=len(groups), desc="Video Types", leave=False):
                Video_Type[vtype] = []
                for i in judge_index:
                    mask = group['index'] == i
                    if not mask.any():
                        continue

                    ans_str = str(group.loc[mask, 'answer'].iloc[0])
                    matches = pattern.findall(ans_str)
                    if len(matches) != 1:
                        continue
                    matched_text = matches[0]
                    pred_str = str(group.loc[mask, 'prediction'].iloc[0])
                    ans_val = matched_text[:-1].lower()
                    correct = pred_str.lower().startswith(ans_val)

                    Video_Type[vtype].append(correct)
            qtype_acc = {k: f"{np.mean([int(x) for x in v]):.3f}" for k, v in Video_Type.items()}
            pth = os.path.join(folder, f"{model}_{noise}_{ratio}_acc_vtype.json")
            print(f"\n=== {model} ACC ===")
            print(json.dumps(qtype_acc, indent=4))
            json.dump(qtype_acc, open(pth, 'w'), indent=4, ensure_ascii=False)


def sbert_score_qtype():
    for model in models:
        for noise in tqdm(noises, desc=f"Model={model} processing sbert"):
            coarse_rating = {k: [] for k in MMV_DIMENSIONS}
            fine_rating = {k: [] for k in L3_DIMS}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            candidates = [f for f in os.listdir(folder) if f.lower().endswith(f'{ratio}.xlsx')][0]
            infile = os.path.join(folder, candidates)
            try:
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue
            for i in trange(len(df), desc="Rows", leave=False):
                ans = str(df.at[i, 'answer'])
                predict = str(df.at[i, 'prediction'])
                cates = eval(str(df.at[i, 'dimensions']))
                emb1 = sbert_model.encode([predict], convert_to_tensor=True, show_progress_bar=False)
                emb2 = sbert_model.encode([ans], convert_to_tensor=True, show_progress_bar=False)
                score = util.cos_sim(emb1, emb2).item()

                for c in cates:
                    fine_rating[c].append(score)

                for d in MMV_DIMENSIONS:
                    if any(x in MMV_DIMENSIONS[d] for x in cates):
                        coarse_rating[d].append(score)

            coarse_sbert = {k: f"{round(np.mean(v), 4)}" for k, v in coarse_rating.items()}
            fine_sbert = {k: f'{round(np.mean(v), 4)}' for k, v in fine_rating.items()}
            rec = dict(coarse_acc=coarse_sbert, fine_acc=fine_sbert)

            pth = os.path.join(folder, f"{model}_{noise}_{ratio}_sbert_qtype.json")
            print(f"\n=== {model} SBERT ===")
            print(json.dumps(rec, indent=4))
            json.dump(rec, open(pth, 'w'), indent=4, ensure_ascii=False)


def sbert_score_vtype():
        for model in models:
            for noise in tqdm(noises, desc=f"Model={model} processing sbert"):
                Video_Type = {}
                folder = os.path.join(ROOT_PATH, model, noise)
                if not os.path.isdir(folder):
                    continue
                candidates = [f for f in os.listdir(folder) if f.lower().endswith(f'{ratio}.xlsx')][0]
                infile = os.path.join(folder, candidates)
                try:
                    df = pd.read_excel(infile)
                except Exception as e:
                    print(f"Error reading {infile}: {e}")
                    continue

                groups = df.groupby(['video_type'])
                for (vtype,), group in tqdm(groups, total=len(groups), desc="Video Types", leave=False):
                    ans = list(group['answer'])
                    predict = list(group['prediction'])
                    emb1 = sbert_model.encode(predict, batch_size=32, convert_to_tensor=True)
                    emb2 = sbert_model.encode(ans, batch_size=32, convert_to_tensor=True)
                    cos_scores = util.cos_sim(emb1, emb2).diag().cpu().tolist()
                    bert_score = round(sum(cos_scores) / len(cos_scores), 3) if cos_scores else None

                    Video_Type[vtype] = bert_score

                pth = os.path.join(folder, f"{model}_{noise}_{ratio}_sbert_vtype.json")
                print(f"\n=== {model} SBERT ===")
                print(json.dumps(Video_Type, indent=4))
                json.dump(Video_Type, open(pth, 'w'), indent=4, ensure_ascii=False)


def gpt_score_qtype():
    for model in models:
        for noise in tqdm(noises, desc=f"Model={model} processing gpt"):
            coarse_rating = {k: [] for k in MMV_DIMENSIONS}
            fine_rating = {k: [] for k in L3_DIMS}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            candidates = [f for f in os.listdir(folder) if f.lower().endswith(f'{ratio}_gpt-4o_score.xlsx')][0]
            infile = os.path.join(folder, candidates)
            try:
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue

            for i in range(len(df)):
                cate = df.iloc[i]['dimensions']
                cates = eval(cate)

                for c in cates:
                    fine_rating[c].append(df.iloc[i]['score'])
                for d in MMV_DIMENSIONS:
                    if np.any([x in MMV_DIMENSIONS[d] for x in cates]):
                        coarse_rating[d].append(df.iloc[i]['score'])
            coarse_gpt = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in coarse_rating.items()}
            fine_gpt = {k: f'{np.mean([max(x, 0) for x in v]):.2f}' for k, v in fine_rating.items()}        

            rec = dict(coarse_gpt=coarse_gpt, fine_gpt=fine_gpt)
            pth = os.path.join(folder, f"{model}_{noise}_{ratio}_gpt_qtype.json")
            print(f"\n=== {model} GPT ===")
            print(json.dumps(rec, indent=4))
            json.dump(rec, open(pth, 'w'), indent=4, ensure_ascii=False)


def gpt_score_vtype():
    for model in models:
        for noise in tqdm(noises, desc=f"Model={model} processing gpt"):
            Video_Type = {}
            folder = os.path.join(ROOT_PATH, model, noise)
            if not os.path.isdir(folder):
                continue
            candidates = [f for f in os.listdir(folder) if f.lower().endswith(f'{ratio}_gpt-4o_score.xlsx')][0]
            infile = os.path.join(folder, candidates)
            try:
                df = pd.read_excel(infile)
            except Exception as e:
                print(f"Error reading {infile}: {e}")
                continue

            groups = df.groupby(['video_type'])
            for (vtype,), group in tqdm(groups, total=len(groups), desc="Video Types", leave=False):
                Video_Type[vtype] = []
                for i in range(len(group)):
                    Video_Type[vtype].append(group.iloc[i]['score'])
            vtype_gpt = {k: f"{np.mean([max(x, 0) for x in v]):.3f}" for k, v in Video_Type.items()}
            pth = os.path.join(folder, f"{model}_{noise}_{ratio}_gpt_vtype.json")
            print(f"\n=== {model} GPT ===")
            print(json.dumps(vtype_gpt, indent=4))
            json.dump(vtype_gpt, open(pth, 'w'), indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Compute metrics: accuracy or SBERT')
    parser.add_argument('--metric', choices=['acc', 'sbert', 'gpt'], required=True,
                        help='Metric to compute: acc or sbert')
    parser.add_argument('--noise', nargs='+', required=True,
                        help='List of noise types to process')
    parser.add_argument('--model', nargs='+', required=True,
                        help='List of model names to process')
    parser.add_argument('--root_path', default='outputs',
                        help='Root directory for demo_output')
    parser.add_argument('--perspective', choices=['qtype', 'vtype'], required=True,
                        help='Analysis across multiple perspectives')
    parser.add_argument('--ratio', type=float, default=None, 
                        help='Proportion of noise frames', choices=[0.3, 0.6, 0.9])
    args = parser.parse_args()

    global noises, models, ROOT_PATH, ratio
    noises = args.noise
    models = args.model
    ROOT_PATH = args.root_path
    ratio = args.ratio
    perspective = args.perspective

    if perspective == 'qtype':
        if args.metric == 'sbert':
            sbert_score_qtype()
        elif args.metric == 'acc':
            accuracy_score_qtype()
        elif args.metric == 'gpt':
            gpt_score_qtype()

    elif perspective == 'vtype':
        if args.metric == 'sbert':
            sbert_score_vtype()
        elif args.metric == 'acc':  
            accuracy_score_vtype()
        elif args.metric == 'gpt':
            gpt_score_vtype()


if __name__ == '__main__':
    main()
