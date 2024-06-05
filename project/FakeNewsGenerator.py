import numpy as np
import os

#open and create dataset

# dataset source: 
#  @article{wang2017liar,
#  title={" liar, liar pants on fire": A new benchmark dataset for fake news detection},
#  author={Wang, William Yang},
#  journal={arXiv preprint arXiv:1705.00648},
#  year={2017}
#  }

data = []

with open("data/train.tsv", "r", encoding= "utf-8") as f:
    train_data = f.readlines()


with open("data/test.tsv", "r", encoding = "utf-8") as f:
    test_data = f.readlines()

with open("data/valid.tsv", "r", encoding = "utf-8") as f:
    valid_data = f.readlines()

sets = [train_data, test_data, valid_data]

for set in sets:
    for statement in set:
        statement = statement.split("\t")[1:3]
        data.append(statement)

data_clean = [statement[1] for statement in data if statement[0] in ["false", "pants-fire", "half-true", "barely-true"]]

#create datasets

print(data_clean[0])



data = " ".join(data_clean)

for spaced in ['.','-',',','!','?','(','—',')']:
    data = data.replace(spaced, ' {0} '.format(spaced))

print(len(data))
#create transition counts

#create transition probabilities

#train model

#create predictions