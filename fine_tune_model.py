import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, GPT2Tokenizer
from datasets import Dataset

data = pd.read_csv("cleaned_dataset.csv")


def preprocess_data(data):
    return Dataset.from_pandas(data)


dataset = preprocess_data(data)

train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(examples):
    inputs = tokenizer(examples['Query'], padding='max_length', truncation=True, max_length=256)
    targets = tokenizer(examples['Result'], padding='max_length', truncation=True, max_length=256)
    inputs['labels'] = targets['input_ids']
    return inputs


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer.train()

model.save_pretrained('./final-fine-tune-bart')
tokenizer.save_pretrained('./final-fine-tune-bart')

