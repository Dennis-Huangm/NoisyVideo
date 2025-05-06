import pandas as pd, os, json
from video_noise import NoiseRegistry

ROOT_PATH = 'outputs'
noises    = ['origin'] + NoiseRegistry.list_noises() 
models = [
    # "Video-LLaVA-7B-HF",
    "VideoChat2-HD",
    # "Chat-UniVi-7B",
    "Chat-UniVi-7B-v1.5",
    "LLaMA-VID-7B",
    # "Video-ChatGPT",
    # "PLLaVA-7B",
    "PLLaVA-13B",
    "Qwen2.5-VL-3B-Instruct",
]
def record_to_csv(eval_dic):
    # 用来统计每个模型 origin 分数低于其他噪声的次数
    issue_counts = {}

    for model in models:
        records = []
        bad_noises = []  # 存放那些比 origin 分数高的噪声

        # 1. 先读取 origin 的 Overall 分数
        origin_overall = None
        try:
            folder_o = os.path.join(ROOT_PATH, model, 'origin')
            fn_o = [f for f in os.listdir(folder_o) if f.endswith('.json')][0]
            data_o = json.load(open(os.path.join(folder_o, fn_o), 'r'))
            origin_overall = data_o['coarse_valid']['Overall']
        except Exception as e:
            print(f"[Warning] 无法读取 {model} origin 数据: {e}")

        # 2. 遍历所有噪声，构造表格行，并做对比
        for noise in noises:
            folder = os.path.join(ROOT_PATH, model, noise)
            try:
                fn = [f for f in os.listdir(folder) if f.endswith('0.9_gpt-4o_rating.json')][0]
                data = json.load(open(os.path.join(folder, fn), 'r'))
            except (IndexError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing {folder}: {e}")
                continue

            valid = data['coarse_valid']  # {'CP':…, 'Overall':…}
            # 按原逻辑检查
            if valid['Overall'] != data['coarse_all']['Overall']:
                print(f"Warning: {model} {noise} Overall 不一致")

            # 只有当读取到 origin_overall 且当前不是 origin 时，做分数比较
            if noise != 'origin' and origin_overall is not None:
                if origin_overall < valid['Overall']:
                    bad_noises.append(noise)

            # 构造表格记录
            if noise == 'origin':
                noise_name = 'clean'
            else:
                noise_name = noise
            # 这里的 noise_type 是为了和原有表格保持一致
            rec = {'noise_type': noise_name}
            rec.update(valid)

            criterions = eval_dic[model].get(noise)
            rec.update(criterions)
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
        # df.to_csv(f'{model}.csv', index=False, encoding='utf-8-sig')

        # 记录本模型统计结果
        issue_counts[model] = len(bad_noises)
        if bad_noises:
            print(f"[注意] {model} 中 origin 分数低于噪声 {bad_noises}")

    # 3. 最终打印每个模型的统计汇总
    print("\n=== Origin Overall 低于其他噪声的统计汇总 ===")
    for model, cnt in issue_counts.items():
        print(f"{model}: {cnt} 次")

def preprocess(df):
    # 原有列顺序调整逻辑不变
    col_to_move = df.pop('Perception')
    pos  = df.columns.get_loc('HL') + 1
    df.insert(pos, 'Perception', col_to_move)

    col_to_move = df.pop('Reasoning')
    pos  = df.columns.get_loc('TR') + 1
    df.insert(pos, 'Reasoning', col_to_move)

