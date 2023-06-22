from transformers import PegasusTokenizer
import os
import torch

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

data_folder = "./outputs"
input_ids = []
attention_masks = []
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            tokens = tokenizer(text, truncation=True, padding="longest", max_length=1020, return_tensors="pt")
            input_ids.append(tokens['input_ids'][0])
            attention_masks.append(tokens['attention_mask'][0])
print(f"Some demo inputIds: {input_ids[0]}")
from torch.utils.data import Dataset
class NewsDataset(Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attn_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
dataset = NewsDataset(input_ids, attention_masks)
folder_path = "model"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
path = './model/my_dataset.pt'
torch.save(dataset, path)