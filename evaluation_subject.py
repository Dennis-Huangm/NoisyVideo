import pandas as pd
import os
from tqdm import tqdm
from video_noise import NoiseRegistry

# Configuration
ROOT_PATH = 'outputs'
noises = ['origin'] + NoiseRegistry.list_noises()
models = [
    "VideoChat2-HD",
    "Chat-UniVi-7B-v1.5",
    "LLaMA-VID-7B",
    "PLLaVA-13B",
    "Qwen2.5-VL-3B-Instruct",
]


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
            files = [f for f in os.listdir(folder) if f.lower().endswith('0.9_gpt-4o_score.xlsx')]
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
        
        gpt_score = group['score'].mean()
        records.append({
            'model': model,
            'noise': noise,
            'video_type': vtype,
            'gpt': gpt_score,
        })

    return pd.DataFrame(records)


def print_latex_tables(metrics_df):
    """
    Print only the LaTeX row entries for each model and metric, in noise_type x video_type order.
    """
    metrics = ['gpt']
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
    # metrics_df.to_csv('metrics.csv', index=False)
    print_latex_tables(metrics_df)
