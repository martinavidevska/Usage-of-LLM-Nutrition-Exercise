import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

# Load dataset
data = pd.read_csv("cleaned_dataset.csv")
dataset = Dataset.from_pandas(data)

# Train/validation split
train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

# Load tokenizer/model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Tokenization
def tokenize_function(examples):
    inputs = ["question: " + q for q in examples["Query"]]
    targets = [a for a in examples["Result"]]

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=256)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=False,
    bf16=True,  # works on M1/M2
    report_to="none"
)

# Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save
model.save_pretrained("./fine-tuned-flan-t5-martina")
tokenizer.save_pretrained("./fine-tuned-flan-t5-martina")
