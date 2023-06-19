import pandas as pd
import requests
import json
import os
import re
from bs4 import BeautifulSoup

# Read the JSON data from file
with open('data.json', 'r') as file:
    lines = file.readlines()

# Parse each JSON object and store in a list
data = []
for line in lines:
    json_object = json.loads(line)
    data.append(json_object)

# Create a DataFrame from the JSON data
df = pd.DataFrame(data)

# Print the DataFrame
new_dataframe = df[:1000]
count = 1
for url in new_dataframe['link']:
    response = requests.get(url)
    if response.status_code == 200: 
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ""
        for paragraph in soup.find_all('p'): #The <p> tag defines a paragraph in the webpages
            text += paragraph.text
        text = text.lower()
        text = re.sub(r'\s+',' ',text)
        # Create a folder to store the files if it doesn't exist
        folder_path = "outputs"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save the extracted text data in a file named "dataq.txt"
        file_path = os.path.join(folder_path, "file_"+str(count)+".txt")
        with open(file_path, 'w') as file:
            file.write(text)
        
        print("Text data saved successfully.")
        count += 1
