#データに含まれる各文の感情分析の結果を集計し，ヒストグラムで図示するプログラム
import pandas as pd
from asari.api import Sonar
import matplotlib.pyplot as plt

# CSVファイルを読み込む
file_path = 'data/labeled_sentences.csv'  # 適切なファイルパスに変更してください
data = pd.read_csv(file_path)

# asariの感情分析器のインスタンスを作成
sonar = Sonar()

# 発信者IDごとに感情分析を行い、confidence値を収集する関数
def collect_confidences(sender_data):
    confidences = []
    for text in sender_data['文']:
        result = sonar.ping(text=text)
        for c in result['classes']:
            if c['class_name'] == result['top_class']:
                score = c['confidence']
                break
        if result['top_class'] == 'positive':
            confidences.append(score)
        elif result['top_class'] == 'negative':
            confidences.append(-score)
    return confidences

# 発信者IDごとに感情分析を行い、confidence値を収集
confidences_by_sender = data.groupby('発信者ID').apply(collect_confidences)

# 各発信者IDごとにヒストグラムを描画
for sender_id, confidences in confidences_by_sender.items():
    plt.figure()
    plt.hist(confidences, bins=[i * 0.05 for i in range(-20, 21)], edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Instances')
    plt.title(f'Confidence Distribution for Sender ID: {sender_id}')
    plt.grid(True)
    plt.show()
