import torch
from torch.utils.data import DataLoader, random_split
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_metric
from tqdm import tqdm
import nltk
nltk.download('punkt')
from rouge import Rouge
import numpy as np
import os
from nltk import word_tokenize


tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
model.resize_token_embeddings(len(tokenizer))

from torch.utils.data import Dataset
class NewsDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attn_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
  
path = './model/my_dataset.pt'
dataset = torch.load(path)

train_size = int(0.8 * len(dataset))

train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('device')

torch.backends.cudnn.enabled = True
learning_rate = 5e-5



training_args = TrainingArguments(output_dir=f'./results_{learning_rate}',
                                  num_train_epochs=1,
                                  logging_steps=1000,
                                  save_steps=1000,
                                  evaluation_strategy='steps',
                                  eval_steps=1000,
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  warmup_steps=100,
                                  learning_rate=learning_rate,
                                  weight_decay=0.01,
                                  gradient_accumulation_steps=500,
                                  logging_dir=f'./logs_{learning_rate}')

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_data,
                  eval_dataset=val_data,
                  # This custom collate function is necessary
                  # to built batches of data
                  data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
              'attention_mask': torch.stack([f[1] for f in data]),
              'labels': torch.stack([f[0] for f in data])})

# Start training process!
print(f"Training result for learning rate: {learning_rate}")
trainer.train()
print("\n\n")

def rougue_score(input_ids, attention_mask):
  text = tokenizer.batch_decode([input_ids], attention_mask=[attention_mask], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
  words = word_tokenize(text)
  length_of_text = len(words)
  summary_ids = model.generate(torch.tensor([input_ids]).to(device), min_length=int(0.20*length_of_text), max_length=int(0.5*length_of_text))
  summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
  print(f"Summary -> {summary}")
  rouge = Rouge()
  score = rouge.get_scores(summary, text)
  return(score[0]['rouge-l']['f'])

training_rouge_score = []
for item in train_data:
  input_ids = item[0].tolist()
  attention_mask = item[1].tolist()
  r_score = rougue_score(input_ids, attention_mask)
  training_rouge_score.append(r_score)
avg_training_score = np.mean(training_rouge_score)
testing_rouge_score = []
for item in val_data:
  input_ids = item[0].tolist()
  attention_mask = item[1].tolist()
  r_score = rougue_score(input_ids, attention_mask)
  testing_rouge_score.append(r_score)
avg_testing_score = np.mean(testing_rouge_score)

print(f"Model's Rouge score on training data is {avg_training_score} and testing data is {avg_testing_score}")
path = "./model"
trainer.save_model(path)
tokenizer.save_pretrained(path)