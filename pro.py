import os
import json
import pandas as pd
from nltk.tokenize import word_tokenize


def delete_short_text_json(folder_path, min_length):
    num=0
    lili=[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            num+=1
            print(num)
    # print(num)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            tmp=data['text'].split()
            # num+=len(data['text'])
            tokens = word_tokenize(data['text'])
            
            if 'text' in data and len(tokens) < min_length:
                # os.remove(file_path)
                print(f"Deleted: {filename}")
            else:
                lili.append(len(tokens))

    return lili

series=delete_short_text_json('./after_pro', 2000)  # 
print(series)
series = pd.Series(series)
mean = series.mean()  # 
median = series.median()  # 
std = series.std()  # 
min_val = series.min()  # 
max_val = series.max()  # 
print(mean,median,std,min_val,max_val)
