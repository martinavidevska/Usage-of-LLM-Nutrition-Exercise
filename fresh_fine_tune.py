from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd

# Load dataset
data = pd.read_csv("cleaned_dataset.csv")
dataset = Dataset.from_pandas(data)

train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

# Load from base BART, not your previous checkpoints
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(examples):
    inputs = ["question: " + str(q) for q in examples["Query"]]
    targets = [str(a) for a in examples["Result"]]  # 👈 ensure all are strings

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=256)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./results_fresh",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=2,  # keep small for MacBook Air
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=False,  # disable mixed precision for CPU/MPS
    bf16=True,  # ok for Apple Silicon
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./fresh-fine-tuned-bart")
tokenizer.save_pretrained("./fresh-fine-tuned-bart")
