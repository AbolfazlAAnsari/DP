from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict
import torch
from torch import nn
from typing import Any, Dict, List
from transformers import default_data_collator
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd

# Local CSV path
local_dataset_path = './data/Contextual_PII_Masking_Dataset.csv'

# Models to fine-tune
model_names = [
    # 't5-small',
    # 't5-base',
    'google/flan-t5-base',
    # 'facebook/bart-base',
]

# Preprocessing function
def get_preprocessor(tokenizer):
    def preprocess(example):
        input_text = 'Mask sensitive information: ' + example['original_text']
        target_text = example['target_text']

        input_enc = tokenizer(input_text, truncation=True, padding='max_length', max_length=128)
        target_enc = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)

        input_ids = input_enc['input_ids']
        attention_mask = input_enc['attention_mask']
        labels = target_enc['input_ids']
        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    return preprocess

# Accuracy metric function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Handle labels with value -100 (ignore during decoding)
    labels = np.where(labels == -100, 0, labels)  # Replace -100 with 0 for decoding

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("Predictions: ", decoded_preds[:5])
    print("Labels: ", decoded_labels[:5])

    correct = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)])
    accuracy = correct / len(decoded_preds)
    return {'accuracy': accuracy}

# Custom Trainer with optional penalty in the loss function
class PrivacyAwareTrainer(Seq2SeqTrainer):
    def __init__(self, penalty=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty  # Add penalty flag

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs['labels']
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for proper loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        # Active tokens to ignore padding in the loss calculation
        active_tokens = (shift_labels != -100).float()
        final_loss = (loss * active_tokens).sum() / active_tokens.sum()

        # Apply penalty to the loss if the flag is set
        if self.penalty:
            penalty_weight = 5.0  # Set your desired penalty weight
            penalty_term = torch.sum(torch.abs(shift_labels.float()))
            final_loss = final_loss + penalty_weight * penalty_term

        return (final_loss, outputs) if return_outputs else final_loss

# Load local dataset
df = pd.read_csv(local_dataset_path)
df = df.dropna(subset=['original_text', 'target_text'])

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert the dataframes to datasets
train_dataset = DatasetDict({'train': Dataset.from_pandas(train_df)})
test_dataset = DatasetDict({'test': Dataset.from_pandas(test_df)})

# Fine-tuning loop for both normal loss and penalized loss
for model_name in model_names:
    print(f'\nFine-tuning model: {model_name} on local dataset')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess datasets
    preprocessor = get_preprocessor(tokenizer)
    train_dataset = train_dataset['train'].map(preprocessor, remove_columns=['original_text', 'target_text'])
    test_dataset = test_dataset['test'].map(preprocessor, remove_columns=['original_text', 'target_text'])

    output_dir = f'./results-local-{model_name.replace("/", "-")}'

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=1,
        save_total_limit=1,
        save_steps=500,
        predict_with_generate=True,
        report_to='none'
    )

    # Scenario 1: Fine-tuning with normal loss
    print(f'Fine-tuning with normal loss for {model_name} on local dataset')
    trainer_normal_loss = PrivacyAwareTrainer(
        penalty=False,  # No penalty for normal loss
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics  # Add accuracy computation
    )
    
    trainer_normal_loss.train()
    trainer_normal_loss.save_model(os.path.join(output_dir, 'normal-loss-final'))

    eval_results_normal = trainer_normal_loss.evaluate()
    print(f'Accuracy with normal loss: {eval_results_normal["eval_accuracy"]:.4f}')



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Encode input and move to the selected device
    input_enc = tokenizer(
        "Emerson Carter's enrollment in the stress and anxiety management program at the counseling center began last fall.",
        return_tensors="pt"
    ).to(device)
    # Move model to the selected device
    model.to(device)
    # Generate output
    generated_ids = model.generate(input_enc['input_ids'], max_length=128, num_beams=4, early_stopping=True)
    # Decode and print
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")



    # Scenario 2: Fine-tuning with penalty on loss
    print(f'Fine-tuning with penalty on loss for {model_name} on local dataset')
    trainer_penalized_loss = PrivacyAwareTrainer(
        penalty=True,  # Penalty applied in the loss function
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics  # Add accuracy computation
    )

    trainer_penalized_loss.train()
    trainer_penalized_loss.save_model(os.path.join(output_dir, 'penalized-loss-final'))

    eval_results_penalized = trainer_penalized_loss.evaluate()
    print(f'Accuracy with penalized loss: {eval_results_penalized["eval_accuracy"]:.4f}')

