import math
import numpy as np

# Keras (for the neural nets)
from keras.models import *
from keras.layers import *
from keras.optimizers import *

vocab = [
    'country',
    'father',
    'king',
    'man',
    'mother',
    'president',
    'queen',
    'woman',
]

VECTOR_SIZE = int(math.log(len(vocab),2)+2)

word_vecs = {
    w : np.random.normal(0,1,size=(VECTOR_SIZE,)) for w in vocab
}

print('Dictionary:')
for w in word_vecs:
    print(w, ':', word_vecs[w])

# Build the data
data = [
    ('country', 'king', 0.5),
    ('country', 'president', 0.5),
    ('country', 'queen', 0.5),
    ('father', 'man', 1.),
    ('king', 'man', 0.1),
    ('king', 'president', -1.),
    ('man', 'woman', -1.),
    ('mother', 'father', -1),
    ('mother', 'woman', 1.),
    ('queen', 'woman', 0.1),
]



def word_filter():
    
    model = Sequential([
        Dense(VECTOR_SIZE, input_shape=(VECTOR_SIZE,))
    ])
    
    model.summary()
    
    return model

def word_model(x1, x2):
    # Define a filter for the words.
    op = word_filter();

    left = Model(inputs=x1, outputs=op(x1))
    right = Model(inputs=x2, outputs=op(x2))
    
    # Compute the cosine.
    y = Merge([left, right], mode='cos')

    # Build the model
    model = Sequential()
    model.add(y)
    model.compile(optimizer=Adagrad(lr=0.1), loss='mean_squared_error')

    model.summary()

    return model

def predict(model, vocab, A, B):
    return model.predict([
        np.array([vocab[A]]),
        np.array([vocab[B]])
    ])[0][0][0]

a = Input(shape=(VECTOR_SIZE,), name='A')
b = Input(shape=(VECTOR_SIZE,), name='B')

model = word_model(a,b)

# Preprocessing
X_left = np.array([word_vecs[x[0]] for x in data])
X_right = np.array([word_vecs[x[1]] for x in data])
Y = np.array([[d[2]] for d in data])

print(X_left.shape)
print(X_right.shape)
print(Y.shape)

model.fit([X_left, X_right], Y, epochs=8192, verbose=0)
for i in range(len(vocab)):
    for j in range(i+1, len(vocab)):
        a, b = vocab[i], vocab[j]
        print('Correlation btwn', a, 'and', b, 'is', predict(model, word_vecs, a, b))



