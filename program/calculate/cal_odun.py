#客観的記述の固有度算出のプログラム
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

# 各固有表現の使用頻度を記録する辞書
entity_frequency = defaultdict(int)

# '発信者ID' ごとに固有表現を格納する辞書
ner_results_by_sender = {}

# 各 '発信者ID' に対して固有表現抽出を実行（進捗バー付き）
for sender_id, group in tqdm(filtered_data.groupby('発信者ID'), desc="Processing"):
    ner_results = []
    for text in tqdm(group['文'], desc=f"Processing texts for sender {sender_id}"):
        entities = ner_pipeline(text)
        for entity in entities:
            entity_text = entity['word']
            entity_frequency[entity_text] += 1
            ner_results.append(entity_text)
    ner_results_by_sender[sender_id] = ner_results

# 各発信者の固有表現被覆度を計算
coverage_scores = {}
for sender_id, entities in ner_results_by_sender.items():
    score = np.average([1.0 / entity_frequency[entity] for entity in entities])
    coverage_scores[sender_id] = score

# 結果を表示
for sender_id, score in coverage_scores.items():
    print(f"発信者ID: {sender_id}, 非被覆度: {score}")
