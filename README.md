# NLP Project - Fine-tuning Google's Hugging Face Pegasus Model

## Summary
This project focuses on fine-tuning the Google's Hugging Face Pegasus model, a state-of-the-art model for abstractive text summarization. The objective is to train the model on a specific dataset and evaluate its performance using the ROUGE score, which measures the quality of the generated summaries. The project consists of several stages, including data preprocessing, tokenization, model selection, and training.

## Setup
To set up the project environment, run the following command:
1. Install the required libraries by running the following command
`!pip install -r requirement.txt`

## Folder Structure
The project's folder structure is as follows:
- data.json
- PreProcessData.py
- TokenizeData.py
- Bestmodel.py
- Train_model.py
- model/
  - [trained models]
- output/
  - [processed data files]
- test/
  - [text files for model evaluation]

Here is a brief description of each file and folder:

### data.json
This file contains the downloaded data from Kaggle, specifically the news category dataset. It provides a quick summary of the dataset, which can be found at the following link: https://www.kaggle.com/datasets/rmisra/news-category-dataset.

### PreProcessData.py
The PreProcessData.py script utilizes the data.json file to download the data from the URLs mentioned in the dataset. It then preprocesses the news articles and saves them as text files in the output/ folder.

### TokenizeData.py
The TokenizeData.py file uses the PegasusTokenizer to convert the preprocessed text into numerical tokens. It then creates a dataset called my_dataset.pt and saves it in the model/ folder. These numerical representations will be used during the model training phase.

### Bestmodel.py
The Bestmodel.py script employs the Hugging Face summarization models and the ROUGE score to identify the best-performing model. It takes text files from the test/ folder as input and evaluates each model's summaries, comparing them against the reference summaries. The script then determines and outputs the best model based on the ROUGE score.

### Train_model.py
The Train_model.py file performs the actual fine-tuning of the Pegasus model. It uses the dataset from my_dataset.pt to train the model, saving the trained model into the model/ folder. Additionally, it evaluates the model's performance using the ROUGE score.

### Website:
Demo website and code can be found at: https://replit.com/@PriyanshZalavad/NewsNebulaFinal
Website shows how to consume model in website using Flask, Python, Html and CSS

## Conclusion
This project showcases the process of fine-tuning Google's Hugging Face Pegasus model for abstractive text summarization. By following the outlined steps, including data preprocessing, tokenization, model selection, and training, it becomes possible to generate high-quality summaries for various text inputs.

Feel free to explore and modify the project to suit your specific needs. Happy summarizing!

