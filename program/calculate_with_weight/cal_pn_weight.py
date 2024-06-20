#肯定度算出プログラム（重み付けあり）
import pandas as pd
from asari.api import Sonar  # asariのSonarクラスをインポート
from tqdm import tqdm  # tqdmモジュールをインポート

# CSVファイルを読み込む
file_path = 'data/labeled_sentences.csv'
data = pd.read_csv(file_path)
filtered_data = data[data['分類ラベル'] == 1]

# asariの感情分析器のインスタンスを作成
sonar = Sonar()

# 発信者ごとの加重平均肯定度を計算する関数
def calculate_weighted_average_positivity(sender_data):
    total_weighted_positivity = 0
    total_weights = 0
    for _, row in tqdm(sender_data.iterrows(), desc='Analyzing Sentiments', total=sender_data.shape[0]):
        text = row['文']
        weight = row['重み']
        result = sonar.ping(text=text)
        top_class = result['top_class']
        score = 0
        for c in result['classes']:
            if c['class_name'] == top_class:
                score = c['confidence']
                break
        if top_class == 'positive':
            total_weighted_positivity += score * weight
        elif top_class == 'negative':
            total_weighted_positivity -= score * weight
        total_weights += weight
    return total_weighted_positivity / total_weights if total_weights > 0 else 0

# 発信者IDごとに感情分析を行い、加重平均肯定度を計算
average_weighted_positivity_scores = filtered_data.groupby('発信者ID').apply(calculate_weighted_average_positivity).reset_index(name='加重平均肯定度')

print(average_weighted_positivity_scores)