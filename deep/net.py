import sys
argv = sys.argv

import numpy as np

# Keras (for the neural nets)
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

# Matplotlib
import matplotlib.pyplot as plt

def one_layer_logic_net(activ=None, opt=Adagrad, loss='mean_squared_error', lr=0.1):
    model = Sequential()
    model.add(Dense(1, input_shape=(2,), activation=activ))
    
    model.compile(optimizer=opt(lr=lr), loss=loss, metrics=['accuracy'])

    return model

def two_layer_logic_net(activ=None, opt=Adagrad, loss='mean_squared_error', lr=0.1):
    model = Sequential()
    model.add(Dense(3, input_shape=(2,), activation=activ))
    model.add(Dense(1, activation=None))
    
    model.compile(optimizer=opt(lr=lr), loss=loss, metrics=['accuracy'])

    return model


LOGIC_DATA_X = np.array([
    [0.,0.], [0.,1.], [1.,0.], [1.,1.]
])

LOGIC_DATA_YS = {
    'AND' : np.array([[0.], [0.], [0.], [1.]]),
    'XOR' : np.array([[0.], [1.], [1.], [0.]])
}
LOGIC_DATA_AND = np.array([
])

LOGIC_DATA_XOR = np.array([
])

net_constructors = {
    'single' : one_layer_logic_net,
    'multi' : two_layer_logic_net
}

activations = {
    'linear' : None,
    'sigmoid' : 'sigmoid',
    'relu' : 'relu'
}

optimizers = {
    'sgd' : SGD,
    'adagrad' : Adagrad,
}

losses = {
    'square' : 'mean_squared_error',
    'cross' : 'binary_crossentropy'
}


if __name__ == '__main__':
    
    if len(argv) < 2:
        print('usage: python', argv[0], 'demo mode type activ opt less')
        print('demos:')
        for k in LOGIC_DATA_YS:
            print(' ', k)

        print('modes:\n  report\n  live')

        print('types:')
        for k in net_constructors:
            print(' ', k)

        print('activations:')
        for k in activations:
            print(' ', k)

        print('optimizers:')
        for k in optimizers:
            print(' ', k)

        print('loss funcs:')
        for k in losses:
            print(' ', k)

        exit()


    demo_name = argv[1]
    

    print('Running', demo_name, 'demo')

    X = LOGIC_DATA_X
    Y = LOGIC_DATA_YS[demo_name]

    demo_mode = argv[2]
    
    # Build the model
    model = net_constructors[argv[3]](activ=activations[argv[4]], opt=optimizers[argv[5]], loss=losses[argv[6]], lr=0.5)
    
    # Report the structure
    print('Using the following structure:')
    model.summary()
    print()

    if demo_mode == 'report':

        print('Running in report mode')

        history = model.fit(X, Y, epochs=8192)
        
        for i in range(len(X)):
            print(X[i][0], demo_name, X[i][1], "=", model.predict(np.array([X[i]]))[0][0])
        
        plt.figure(1)
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        
        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        plt.show()
    
    elif demo_mode == 'live':

        print('Running in live mode')

        started = False

        iters = 0
        
        plt.ion()
        while True:
            model.fit(X,Y, epochs=1, verbose=0)
            iters += 1

            xs = np.arange(-0.5, 1.5, 0.05)
            ys = np.flip(np.arange(-0.5, 1.5, 0.05), axis=0)

            Z = np.array([
                [
                    model.predict(np.array([[i, j]]))[0][0]
                    for j in xs
                ] for i in ys
            ])

            plt.imshow(Z, cmap='seismic', interpolation='nearest', extent=[-0.5,1.5,-0.5,1.5], vmin=0, vmax=1)

            plt.title(str(demo_name) + ' Output Map (iterations: ' + str(iters) + ")")
            
            if not started:
                # Provide a colorbar
                plt.colorbar()
                started = True

            plt.draw()
            plt.pause(0.01)

    else:
        print('Unrecognized mode', demo_mode)


