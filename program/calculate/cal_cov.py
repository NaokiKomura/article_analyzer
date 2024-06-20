#被覆度算出のプログラム
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 固有表現抽出のパイプラインを初期化
ner_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)

# CSVファイルを読み込む
data = pd.read_csv('data/labeled_sentences.csv')
filtered_data = data[data['分類ラベル'] == 0]

# 全発信者によって使用された固有表現のユニークな集合
unique_entities = set()

# 各発信者の固有表現のユニークな集合を記録する辞書
unique_entities_by_sender = {}

# 各 '発信者ID' に対して固有表現抽出を実行（進捗バー付き）
for sender_id, group in tqdm(filtered_data.groupby('発信者ID'), desc="Processing"):
    sender_entities = set()
    for text in tqdm(group['文'], desc=f"Processing texts for sender {sender_id}"):
        entities = ner_pipeline(text)
        for entity in entities:
            entity_text = entity['word']
            sender_entities.add(entity_text)
            unique_entities.add(entity_text)
    unique_entities_by_sender[sender_id] = sender_entities

# 各発信者の固有表現の割合を計算
entity_ratios = {sender_id: len(entities) / len(unique_entities) for sender_id, entities in unique_entities_by_sender.items()}

# 結果を表示または別の方法で処理
for sender_id, ratio in entity_ratios.items():
    print(f"発信者ID: {sender_id}, 固有表現の割合: {ratio}")
