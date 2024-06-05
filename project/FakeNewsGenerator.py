import numpy as np
import os
from scipy.sparse import dok_matrix
import random
from random import random

#open and create dataset

# dataset source: 
#  @article{wang2017liar,
#  title={" liar, liar pants on fire": A new benchmark dataset for fake news detection},
#  author={Wang, William Yang},
#  journal={arXiv preprint arXiv:1705.00648},
#  year={2017}
#  }

def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects', 
        the likelihood of the objects is weighted according 
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]


def sample_next_word_after_sequence(word_sequence, alpha = 0):
    next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]] + alpha
    likelihoods = next_word_vector/next_word_vector.sum()

    return weighted_choice(data_distinct, likelihoods.toarray())





if __name__ == "__main__":

    data = []

    with open("data/train.tsv", "r", encoding= "utf-8") as f:
        train_data = f.readlines()


    with open("data/test.tsv", "r", encoding = "utf-8") as f:
        test_data = f.readlines()

    with open("data/valid.tsv", "r", encoding = "utf-8") as f:
        valid_data = f.readlines()

    sets = [train_data, test_data, valid_data]

    for sentences in sets:
        for statement in sentences:
            statement = statement.split("\t")[1:3]
            data.append(statement)

    data_clean = [statement[1] for statement in data if statement[0] in ["false", "pants-fire", "half-true", "barely-true"]]

    #create datasets

    print(data_clean[0])



    data = " ".join(data_clean)

    for spaced in ['.','-',',','!','?','(','—',')']:
        data = data.replace(spaced, ' {0} '.format(spaced))

    #collect token and word counts

    data_words = data.split(" ")

    data_tokens = [word for word in data_words if word != ""]

    print(len(data_tokens))

    data_distinct = list(set(data_tokens))
    word_idx_dict = {word: i for i, word in enumerate(data_distinct)}

    print(len(data_distinct))



    #train model

    k = 2

    sets_of_k_words = [" ".join(data_tokens[i:i+k]) for i, _ in enumerate(data_tokens[:-k])]

    sets_count = len(list(set(sets_of_k_words)))
    next_after_k_words_matrix = dok_matrix((sets_count, len(data_distinct)))

    distinct_set_of_k_words = list(set(sets_of_k_words))
    k_words_idx_dict = {word:i for i, word in enumerate(distinct_sets_of_k_words)}

    for i, word in enumerate(sets_of_k_words[:-k]):
        word_sequence_idx = k_words_idx_dict[word]
        next_word_idx = word_idx_dict[data_tokens[i+k]]
        next_after_k_words_matrix[word_sequence_idx, next_word_idx] +=1
    
    #create predictions

    