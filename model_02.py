from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import default_data_collator
import numpy as np
import os

# Models to fine-tune
model_names = [
    't5-small',
    't5-base',
    'google/flan-t5-base',
    'facebook/bart-base',
]

# Load the dataset
dataset = load_dataset('ai4privacy/pii-masking-65k')

# Preprocessing function
def get_preprocessor(tokenizer):
    def preprocess(example):
        # Input: unmasked text
        input_text = 'Mask sensitive information: ' + example['unmasked_text']
        
        # Target: masked text
        target_text = example['masked_text']
        
        # Tokenize inputs and targets
        input_enc = tokenizer(
            input_text, 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
        
        target_enc = tokenizer(
            target_text, 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
        
        # Prepare labels (set padding tokens to -100)
        labels = target_enc['input_ids']
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
        
        return {
            'input_ids': input_enc['input_ids'],
            'attention_mask': input_enc['attention_mask'],
            'labels': labels
        }
    return preprocess

# Example: fine-tune one model
model_checkpoint = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Preprocess dataset
processed_dataset = dataset.map(
    get_preprocessor(tokenizer),
    batched=True,
    remove_columns=dataset['train'].column_names, # remove original columns
)

# Training arguments
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

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Train
trainer.train()
