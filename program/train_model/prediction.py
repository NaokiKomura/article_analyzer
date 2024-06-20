from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import pandas as pd
import torch

# モデルとトークナイザの読み込み
model_name = "naokikomura/deberta-v2-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 分類パイプラインの設定
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# データの読み込み
file_path = 'data/cleaned_converted_validation_set.csv'
df = pd.read_csv(file_path)

# 各文に対して分類を行う関数
def classify_sentence(sentence):
    result = pipeline(sentence)
    label = result[0]['label']
    return 1 if label == 'LABEL_1' else 0

# 分類ラベルの付与
df['分類ラベル'] = df['文'].apply(classify_sentence)

# 結果のCSVとして保存
output_path = 'data/classified_csv'
df.to_csv(output_path, index=False)
