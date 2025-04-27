import pandas as pd, os, json
from video_noise import NoiseRegistry

ROOT_PATH = 'outputs'
noises    = NoiseRegistry.list_noises() + ['origin']  
models    = ['Video-LLaVA-7B-HF', 'LLaMA-VID-7B', 'VideoChat2-HD', 'Chat-UniVi-7B', "Chat-UniVi-7B-v1.5", "Video-ChatGPT", "PLLaVA-7B"]

def record_to_csv():
    """
    读取每个模型的每种噪声的指标，拼成一个表格
    :return:
    """
    # 读取每个模型的每种噪声的指标，拼成一个表格
    # 每个模型一个表格
    for model in models:
        records = []
        for noise in noises:
            # 读 JSON，拿出你想要的指标字典
            folder = os.path.join(ROOT_PATH, model, noise)
            # 找到第一个 .json
            try:
                fn = [f for f in os.listdir(folder) if f.endswith('.json')][0]
                data = json.load(open(os.path.join(folder, fn), 'r'))
            except (IndexError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing {folder}: {e}")
                continue
            valid = data['coarse_valid']     # e.g. {'CP':0.8, 'FP-S':0.1, …}
            if data['coarse_valid']['Overall'] != data['coarse_all']['Overall']:
                print(f"Warning: {model} {noise} Overall not equal")

            # 构造这一行
            rec = {'noise_type': noise}
            rec.update(valid)                # rec 里就有噪声名 + 各指标
            records.append(rec)

        # 把所有噪声的记录拼成一个表
        df = pd.DataFrame(records)
        preprocess(df)
        print(model)
        print(df)
        cols = df.columns.tolist()
        cols.pop(0)  # 去掉第一列

        # for _, row in df.iterrows():
        #     name = row['noise_type'].replace('_', '\\_')       # 下划线要转义
        #     vals = ' & '.join(f"{row[c]}" for c in cols)
        #     print(f"{name} & {vals} \\\\")
        # df.to_csv(f'{model}.csv', index=False, encoding='utf-8-sig')


def preprocess(df):
    col_to_move = df.pop('Perception')
    pos  = df.columns.get_loc('HL') + 1
    df.insert(pos, 'Perception', col_to_move)

    col_to_move = df.pop('Reasoning')
    pos  = df.columns.get_loc('TR') + 1
    df.insert(pos, 'Reasoning', col_to_move)


if __name__ == '__main__':
    record_to_csv()
