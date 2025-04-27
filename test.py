from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Load tokenizer and model (pre-trained version)
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Load the fine-tuned model (based on the English dataset)
fine_tuned_model_path = '/Users/abolfazlansari/Documents/Pike/Code/DP/DP/results-pii-masking-400k-t5-small-full/normal-loss-final'
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path)

# 3. Function to make predictions using the model
def predict(model, sentence):
    # Prepend a task-specific tag (e.g., "Mask sensitive information") instead of translation tag
    sentence_with_tag = "Mask sensitive information: " + sentence
    inputs = tokenizer(sentence_with_tag, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=128)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# 4. Test sentence
test_sentence = """ My name is Abolfazl Ansari and I am a software engineer. My email is test@gmail.com and my phone number is 123-456-7890. I live in Tehran, Iran. I have weekly visit to hospital"""
pretrained_prediction = predict(pretrained_model, test_sentence)
print("Pre-trained model prediction:", pretrained_prediction)

# 6. Make predictions with the fine-tuned model (trained only on English data)
fine_tuned_prediction = predict(fine_tuned_model, test_sentence)
print("Fine-tuned model prediction:", fine_tuned_prediction)

