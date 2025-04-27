from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import load_dataset
import numpy as np
import os

# Model names to fine-tune
model_names = [
    't5-small',
    't5-base',
    'google/flan-t5-base',
    'facebook/bart-base',
]

# Load the dataset
dataset = load_dataset('ai4privacy/pii-masking-65k')

# Manually split the training set
split_dataset = dataset['train'].train_test_split(test_size=0.1)

# Preprocessing function
def get_preprocessor(tokenizer):
    def preprocess(example):
        input_text = example['unmasked_text']
        if isinstance(input_text, list):
            input_text = ' '.join(input_text)

        target_text = example['masked_text']
        if isinstance(target_text, list):
            target_text = ' '.join(target_text)

        inputs = tokenizer(
            'Mask sensitive information: ' + input_text,
            truncation=True,
            padding='max_length',
            max_length=512
        )

        targets = tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=512
        )

        labels = targets['input_ids']
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }
    return preprocess

# Fine-tune one model
model_checkpoint = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Preprocess the split dataset
processed_dataset = split_dataset.map(
    get_preprocessor(tokenizer),
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc='Tokenizing dataset',
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f'./results/{model_checkpoint}',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=100,
    push_to_hub=False,
    report_to='none'
)

# Define Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],  # Use 'test' as validation split
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Start training
trainer.train()
