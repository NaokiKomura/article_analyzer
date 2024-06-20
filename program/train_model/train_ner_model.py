from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import set_seed
from spacy_alignments.tokenizations import get_alignments
import torch
from huggingface_hub import login


dataset = load_dataset("llm-book/ner-wikipedia-dataset")

model_name = "ku-nlp/deberta-v2-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
'''
subwords = "/".join(tokenizer.tokenize(dataset["train"][0]["text"]))
characters = "/".join(dataset["train"][0]["text"])
print(f"サブワード単位：{subwords}")
print(f"文字単位:{characters}")
'''

def create_label2id(
        entities_list: list[list[dict[str, str | int]]]
) -> dict[str, int]:
    """ラベルとIDを紐付けるdictを作成"""
    label2id = {"0":0}
    entity_types = set(
        [e["type"] for entities in entities_list for e in entities]
    )
    entity_types = sorted(entity_types)
    for i, entity_types in enumerate(entity_types):
        label2id[f"B-{entity_types}"] = i * 2+1
        label2id[f"I-{entity_types}"] = i * 2+2
    return label2id

label2id = create_label2id(dataset["train"]["entities"])
id2label = {v: k for k, v in label2id.items()}


def preprocess_data(
        data:dict[str, any],
        tokenizer:PreTrainedTokenizer,
        label2id:dict[int, str],
)-> BatchEncoding:
    """データの前処理"""
    inputs = tokenizer(
        data["text"],
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    inputs = {k:v.squeeze(0) for k, v in inputs.items()}

    characters = list(data["text"])
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    char_to_token_indices, _ = get_alignments(characters, tokens)

    labels = torch.zeros_like(inputs["input_ids"])
    for entity in data["entities"]:
        start_token_indices = char_to_token_indices[entity["span"][0]]
        end_token_indices = char_to_token_indices[
            entity["span"][1] - 1
        ]

        if(
            len(start_token_indices) == 0
            or len(end_token_indices) == 0
        ):
            continue
        start, end = start_token_indices[0], end_token_indices[0]
        entity_type = entity["type"]

        labels[start] = label2id[f"B-{entity_type}"]
        if start != end:
            labels[start + 1: end + 1] = label2id[f"I-{entity_type}"]

    labels[torch.where(inputs["special_tokens_mask"])] = -100
    inputs["labels"] = labels
    return inputs

train_dataset = dataset["train"].map(
    preprocess_data,
    fn_kwargs={
        "tokenizer":tokenizer,
        "label2id":label2id,
    },
    remove_columns=dataset["train"].column_names,
)

validation_dataset = dataset["validation"].map(
    preprocess_data,
    fn_kwargs={
        "tokenizer":tokenizer,
        "label2id":label2id
    },
    remove_columns=dataset["validation"].column_names
)

model = AutoModelForTokenClassification.from_pretrained(
    model_name, label2id=label2id, id2label=id2label
)

data_collator = DataCollatorForTokenClassification(tokenizer)

set_seed(42)

training_args = TrainingArguments(
    output_dir="output_deberta_ner",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch", 
)

trainer = Trainer(
    model = model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    args=training_args,
)

trainer.train()

login()
repo_name = "naokikomura/deberta-base-japanese-v2-ner"

tokenizer.push_to_hub(repo_name)
model.push_to_hub(repo_name)

