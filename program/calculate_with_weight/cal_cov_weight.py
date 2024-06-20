#被覆度算出プログラム（重み付けあり）
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

# 各固有表現の重みを格納する辞書
weights_for_entities = {}

# 各 '発信者ID' に対して固有表現抽出を実行
for sender_id, group in tqdm(filtered_data.groupby('発信者ID'), desc="Processing"):
    sender_entities = set()
    for _, row in group.iterrows():
        text = row['文']
        weight = row['重み']
        entities = ner_pipeline(text)
        for entity in entities:
            entity_text = entity['word']
            sender_entities.add(entity_text)
            unique_entities.add(entity_text)

            # 各固有表現に対する重みのリストを更新
            if entity_text not in weights_for_entities:
                weights_for_entities[entity_text] = []
            weights_for_entities[entity_text].append(weight)
            
    unique_entities_by_sender[sender_id] = sender_entities

# 固有表現の重みの平均を計算
average_weights_for_entities = {entity: sum(weights) / len(weights) for entity, weights in weights_for_entities.items()}

# 発信者ごとの固有表現の重みの合計を計算し、割合を算出
total_weight_of_all_entities = sum(average_weights_for_entities.values())
entity_ratios_by_sender = {}
for sender_id, entities in unique_entities_by_sender.items():
    total_weight_for_sender = sum(average_weights_for_entities[entity] for entity in entities)
    entity_ratios_by_sender[sender_id] = total_weight_for_sender / total_weight_of_all_entities

# 結果の出力（例）
print(entity_ratios_by_sender)
