from pprint import pprint
import pandas as pd
from transformers import BatchEncoding, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import numpy as np
import os
from huggingface_hub import login

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

model_name = "ku-nlp/deberta-v2-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_text_classification(example: dict[str, str | int]) -> BatchEncoding:
    encoded_example = tokenizer(example["sentence"], max_length=512, truncation=True)
    encoded_example["labels"] = example["label"]
    return encoded_example


# CSVファイルのリスト
csv_files = [
    "data/torkenized/tokenized_classification_data_1.csv",
    "data/torkenized/tokenized_classification_data_2.csv",
    "data/torkenized/tokenized_classification_data_3.csv"
]

eval_files = [
    "data/torkenized/tokenized_classification_data_4.csv"
]

# 処理後のデータを格納するためのリスト
processed_data = []
eval_data = []

for file in eval_files:
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        example = {
            "sentence": row["text"],
            "label": row["label"]
        }
        encoded_example = preprocess_text_classification(example)
        eval_data.append(encoded_example)



for file in csv_files:
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        example = {
            "sentence": row["text"],
            "label": row["label"]
        }
        encoded_example = preprocess_text_classification(example)
        processed_data.append(encoded_example)

data_collator = DataCollatorWithPadding(tokenizer= tokenizer)

batch_inputs = data_collator(processed_data[0:4])

num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 訓練設定
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)
   
def compute_accuracy(
        eval_pred: tuple[np.ndarray, np.ndarray]
)-> dict[str, float]:
    """よそ比べると正解ラベルから正解率を計算"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy":(predictions == labels).mean()}

trainer = Trainer(
    model=model,
    eval_dataset=eval_data,
    train_dataset=processed_data,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_accuracy
) 

trainer.train()

eval_metrics = trainer.evaluate(eval_data)
pprint(eval_metrics)

login()
repo_name = "naokikomura/deberta-v2-base-japanese"

tokenizer.push_to_hub(repo_name)
model.push_to_hub(repo_name)

