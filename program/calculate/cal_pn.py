#肯定度の算出プログラム
import pandas as pd
from asari.api import Sonar  # asariのSonarクラスをインポート
from tqdm import tqdm  # tqdmモジュールをインポート

# CSVファイルを読み込む
file_path = 'data/labeled_sentences.csv'
data = pd.read_csv(file_path)

# "分類ラベル"が1のデータのみを選択
filtered_data = data[data['分類ラベル'] == 1]

# asariの感情分析器のインスタンスを作成
sonar = Sonar()

# 発信者ごとの平均肯定度を計算する関数
def calculate_average_positivity(sender_data):
    total_positivity = 0
    message_count = 0
    for text in tqdm(sender_data['文'], desc='Analyzing Sentiments'):
        result = sonar.ping(text=text)
        top_class = result['top_class']
        for c in result['classes']:
            if c['class_name'] == top_class:
                score = c['confidence']
                break
        if top_class == 'positive':
            total_positivity += score
        elif top_class == 'negative':
            total_positivity -= score
        message_count += 1
    return total_positivity / message_count if message_count > 0 else 0

# 発信者IDごとに感情分析を行い、平均肯定度を計算
average_positivity_scores = filtered_data.groupby('発信者ID').apply(calculate_average_positivity).reset_index(name='平均肯定度')

# 結果をファイルに保存または表示
print(average_positivity_scores)
#average_positivity_scores.to_csv('analyzed_filtered_data_with_progress.csv', index=False)
