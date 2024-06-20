#客観的記述の固有度算出プログラム（重み付けあり）
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# 固有表現抽出のパイプラインを初期化
ner_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)

# CSVファイルを読み込む
data = pd.read_csv('data/labeled_sentences.csv')
filtered_data = data[data['分類ラベル'] == 0]

# 発信者IDごとに固有表現の加重頻度と加重非被覆度スコアを格納する辞書
weighted_entity_frequency_by_sender = defaultdict(lambda: defaultdict(float))
weighted_coverage_scores_by_sender = defaultdict(list)

# 各 '発信者ID' に対して固有表現抽出を実行（進捗バー付き）
for sender_id, group in tqdm(filtered_data.groupby('発信者ID'), desc="Processing Senders"):
    for _, row in tqdm(group.iterrows(), desc=f"Processing texts for sender {sender_id}"):
        text = row['文']
        weight = row['重み']
        entities = ner_pipeline(text)
        for entity in entities:
            entity_text = entity['word']
            # 発信者IDごとの重みを考慮した加重頻度の更新
            weighted_entity_frequency_by_sender[sender_id][entity_text] += weight  
            # 発信者IDごとの加重非被覆度スコアの計算
            frequency = weighted_entity_frequency_by_sender[sender_id][entity_text]
            weighted_score = weight / (frequency if frequency > 0 else 1e-10)
            weighted_coverage_scores_by_sender[sender_id].append(weighted_score)

# 各発信者の平均加重非被覆度を計算して表示
for sender_id, scores in weighted_coverage_scores_by_sender.items():
    average_weighted_score = np.mean(scores) if scores else 0
    print(f"発信者ID: {sender_id}, 平均加重非被覆度: {average_weighted_score}")