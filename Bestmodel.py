#In this we will explore how we can find best model using ROGUE score
from transformers import pipeline
from rouge import Rouge
from nltk import word_tokenize
import glob
import numpy as np

models = ['facebook/bart-large-cnn',
          'philschmid/bart-large-cnn-samsum',
          'sshleifer/distilbart-cnn-12-6',
          'moussaKam/barthez-orangesum-abstract',
          'google/pegasus-cnn_dailymail',
          'google/bigbird-pegasus-large-bigpatent',
          'csebuetnlp/mT5_multilingual_XLSum']

final_score = -10
best_model = ''
for model in models:
    summarize = pipeline("summarization", model=model)
    path = 'test/*.txt'

    rouge_score =[]
    for file_name in glob.glob(path):
        with open(file_name, "r") as file:
            data = file.read()
            words = word_tokenize(data)
            length = len(words)
            if length>500:
                words = words[:500]
            length = len(words)
            text = ' '.join(words)
            summary = summarize(text, max_length=int(length*0.50), min_length=int(length*0.20), do_sample=False)[0]["summary_text"]
            rouge = Rouge()
            scores = rouge.get_scores(summary, text)
            rouge_score.append(scores[0]['rouge-l']['f'])
            print(f" Summary of Filename - {file_name}\n Summary - {summary} \n Model - {model} \n Score - {scores[0]['rouge-l']['f']}")
            print("-"*25)
    mean_score = np.mean(rouge_score)
    if mean_score > final_score:
        final_score = mean_score
        best_model = model
print(f"Best model is {best_model} with score {final_score}")




