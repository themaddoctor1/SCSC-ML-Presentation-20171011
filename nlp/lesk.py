import nltk
import itertools
import sys

# Get word definitions
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

def sentence_overlap(sentences):
    return sum(1 if len(set(item)) <= 1 else 0 for item in itertools.product(*sentences))

def my_simple_lesk(word, sentence):
    """Performs the simplified Lesk algorithm.
    word - A string containing the word to check.
    sentence - A string containing a sentence.
    """
    
    # The 'context' is the set of words in the sentence.
    context = nltk.word_tokenize(sentence)

    best_sense = None
    max_overlap = 0
    
    # We check every definition of the word
    for sense in wn.synsets(word):
        # Get the set of words in the sense/definition of the word
        signature = nltk.word_tokenize(sense.definition())

        overlap = sentence_overlap([context, signature])

        
        # If the current sense is deemed to be better, choose it.
        if overlap > max_overlap:
            best_sense = sense.definition()
            max_overlap = overlap

    return best_sense

def simple_lesk(word, sentence):
    return lesk(nltk.word_tokenize(sentence), word).definition()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python', sys.argv[0], 'A/B', '<word>', '<sentence>')

    if sys.argv[1] == 'A':
        print('Using my logic implementation')
        f = my_simple_lesk
    elif sys.argv[1] == 'B':
        print('Using the NLTK implementation')
        f = simple_lesk
    else:
        print('Not a valid input')
        exit()

    print('Word:', sys.argv[2])
    print('Sentence:', sys.argv[3])

    print('Definition:', f(sys.argv[2], sys.argv[3]))



