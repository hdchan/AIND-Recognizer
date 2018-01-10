import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for test_word in test_set.wordlist:
        dic = {}
        idx = test_set.wordlist.index(test_word)
        test_X, test_lengths = test_set.get_item_Xlengths(idx)
        best_guess = None
        best_score = float("-inf")
        for current_word in models:
            current_model = models[current_word]
            if (current_word in test_set.wordlist):
                try:
                    score = current_model.score(test_X, test_lengths)
                    dic[current_word] = score
                    if score > best_score:
                        best_score = score
                        best_guess = current_word
                except:
                    continue
        probabilities.append(dic)
        guesses.append(best_guess)

    return probabilities, guesses
