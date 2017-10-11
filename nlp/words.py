import gensim, sys

from nltk import tokenize
from nltk.corpus import stopwords

f = open(sys.argv[1])
text = f.read()
f.close()

sentences = tokenize.sent_tokenize(text)

for s in sentences:
    print(s)

stopWords = set(stopwords.words('english'))
data = [list(filter(lambda w : w not in stopWords, tokenize.word_tokenize(x))) for x in sentences]

for x in data:
    print(x)

model = gensim.models.Word2Vec(data, min_count=1, size=10)

print('Vocabulary:')
for w in model.wv.vocab:
    sys.stdout.write(w + ' ')
print()

model.train(data, total_examples=model.corpus_count, epochs=25)

def print_similarity(model, A, B):
    print(  'Similarity between',
            A, 'and', B, 'is',
            model.similarity(A, B))

def get_word_vector(model, word):
    return model.wv[word]

test = sys.argv[2:]
for i in range(len(test)):
    for j in range(i+1, len(test)):
        print_similarity(model, test[i], test[j])
        #print('Word vector of', sys.argv[i], 'is', get_word_vector(model, sys.argv[i]))
        #print('Word vector of', sys.argv[i+1], 'is', get_word_vector(model, sys.argv[i+1]))

