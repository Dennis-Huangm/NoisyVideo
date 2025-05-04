import pandas as pd, os
from video_noise import NoiseRegistry
import re
from tqdm import tqdm
from to_csv import record_to_csv
import sacrebleu
from bert_score import score
from evaluate import load
from sentence_transformers import SentenceTransformer, util

rouge_judge = load('rouge')
ROOT_PATH = 'outputs'
noises  = ['origin'] + NoiseRegistry.list_noises() 
models  = ["Qwen2.5-VL-3B-Instruct"]
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
dic = {}
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def accuracy_score():
    for model in models:
        dic[model] = {}
        for noise in tqdm(noises):
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
            
            acc_num = 0
            num_judge = 0
            num_digital = 0
            acc_judge = 0
            acc_digital = 0
            for i in index:    
                matches = pattern.findall(str(df.at[i, 'answer']))
                if len(matches) != 1:
                    raise ValueError(f"Row {i} does not match: {matches}")
                else:
                    matches = matches[0]

                predict = str(df.at[i, 'prediction'])

                
                if digital_pattern.match(matches):
                    num_digital += 1    
                    ans = matches
                    if ans in predict:
                        acc_num += 1
                        acc_digital += 1
                        # print('question：' + str(df.at[i, 'question']))
                        # print('ans：' + str(df.at[i, 'answer']))
                        # print('predict：' + predict)
                        # print('\n')
                else:
                    num_judge += 1
                    ans = matches[:-1].lower() if matches[-1] in [',', '.'] else matches
                    if predict.lower().startswith(ans):
                        acc_num += 1
                        acc_judge += 1
                        # print('question：' + str(df.at[i, 'question']))
                        # print('ans：' + str(df.at[i, 'answer']))
                        # print('predict：' + predict)
                        # print('\n')
            dic[model][noise] = {'acc': round(acc_num / len(index), 3), \
                                'acc_judge': round(acc_judge / num_judge, 3), \
                                'acc_digital': round(acc_digital / num_digital, 3)}


def metric_score():
    for model in models:
        for noise in tqdm(noises):
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
            
            ans = []
            predict = []
            for i in range(len(df)):
                ans.append(str(df.at[i, 'answer']))
                predict.append(str(df.at[i, 'prediction'])) 

            bleu_scores = []
            for pred, ref in zip(predict, ans):
                # sacrebleu.sentence_bleu 接受单条 hypothesis（字符串）
                # 和一个 list of references（这里每条只有一个参考）
                sent_bleu = sacrebleu.sentence_bleu(
                    pred,
                    [ref],
                    smooth_method='exp'
                )
                # .score 范围 0–100
                bleu_scores.append(sent_bleu.score)
            print(len(predict), len(ans))
            bleu = round(sum(bleu_scores)/len(bleu_scores), 2)
            print(f"BLEU: {bleu}")
            rouge = round(rouge_judge.compute(predictions=predict, references=ans)['rougeLsum'], 2)
            print(f"ROUGE: {rouge}")
            # bert = round(score(predict, ans, lang='en', verbose=True)[2].mean().item(), 2)
            embeddings1 = sbert_model.encode(
                predict,
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            embeddings2 = sbert_model.encode(
                ans,
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            pairwise_scores = cosine_scores.diag().cpu().tolist()
            bert = round(sum(pairwise_scores)/len(pairwise_scores), 2)
            print(f"BERT: {bert}")

            dic[model][noise].update({'bleu': bleu, 'rouge': rouge, 'bert': bert})


if __name__ == '__main__':
    accuracy_score()
    metric_score()
    print(dic)
    record_to_csv(dic)
    

        