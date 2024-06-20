#主観的記述の固有度算出プログラム
import pandas as pd
from asari.api import Sonar
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sonar = Sonar()

# CSVファイルの読み込み
df = pd.read_csv('data/labeled_sentences.csv')

# 分類ラベルが1のデータの抽出
df_filtered = df[df['分類ラベル'] == 1]

# 結果を保存するための辞書
results = {}

# 発信者IDごと、そしてクラスターラベルごとに処理
for sender_id in df_filtered['発信者ID'].unique():
    sender_df = df_filtered[df_filtered['発信者ID'] == sender_id]
    for cluster_label in range(4):  # クラスターラベルは0から3まで
        cluster_df = sender_df[sender_df['クラスター'] == cluster_label]

        # asariで感情分析（tqdmを使用して進捗を表示）
        sentiments = []
        for text in tqdm(cluster_df['文'], desc=f"Processing Sender ID {sender_id}, Cluster Label {cluster_label}"):
            analysis = sonar.ping(text)
            top_class = analysis["top_class"]
            confidence = [c["confidence"] for c in analysis["classes"] if c["class_name"] == top_class][0]

            # Positiveならそのまま、Negativeなら-1を掛ける
            sentiment_score = confidence if top_class == "positive" else -confidence
            sentiments.append(sentiment_score)

        # 発信者とクラスターラベルごとの結果の和を計算
        results[(sender_id, cluster_label)] = sum(sentiments)

# 各発信者ごとにクラスターラベルを次元とする4次元ベクトルを作成
vectors = {sender_id: [results.get((sender_id, cluster), 0) for cluster in range(4)]
           for sender_id in df_filtered['発信者ID'].unique()}

print(vectors)

# コサイン類似度を計算する関数
def cosine_distance(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# 全発信者間のコサイン類似度の平均を計算
avg_cosine_distances = {}
for sender_id in vectors:
    distances = []
    for other_id, other_vector in vectors.items():
        if sender_id != other_id:
            distance = cosine_distance(vectors[sender_id], other_vector)
            distances.append(distance)
    avg_cosine_distances[sender_id] = np.mean(distances) if distances else 0

# 結果の出力
print("Average Cosine Distances Between Senders:")
for sender_id, avg_distance in avg_cosine_distances.items():
    print(f"Sender ID {sender_id}: {avg_distance}")
