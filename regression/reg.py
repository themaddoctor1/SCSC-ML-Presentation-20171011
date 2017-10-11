import numpy as np
import math, sys

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def linreg_proj(X, y):
    """Computes weights w such that (X^T)*X*w = (X^T)*y
    """
    A = np.matmul(np.transpose(X), X)
    b = np.matmul(np.transpose(X), y)

    return np.linalg.solve(A, b)

def logreg_proj(X, y):
    # We need to project onto the solvable space.
    z = np.vectorize(lambda x : 512. if x == 1.0 else -512. if x == 0.0 else math.log(x / (1.0 - x)))(y)
    
    for i in range(len(z)):
        print(y[i], '=>', z[i])

    return linreg_proj(X, z)

def logreg_sgd(X, Y, alpha = 0.1, epochs=1, w = None, batch=False):
    
    if w is None:
        w = np.random.normal(size=X.shape[1])
    
    if batch:
        W = w * 1.0

    for _ in range(epochs):
        for i in range(len(X)):
            x = X[i]
            t = Y[i][0]

            s = np.dot(w, x)
            y = sigmoid(s)

            de = (y - t) * alpha

            # Backpropagate
            deds = de*sigmoid(s)*(1-sigmoid(s))
            dedw = deds*x # ds/dw = x^T
            
            #print('w :=', w)

            if batch:
                W = W - dedw
            else:
                w = w - dedw
            
            """
            print('Given x :=', x, 'expecting', t, ', got', y)
            print('dE/dy :=', de)
            print('dE/ds :=', deds)
            print('dE/dw :=', dedw) 
            print('w\' :=', w, '\n')
            """

    return W if batch else w


def linreg_demo(m, b, mean, stddev):
    X = np.array([
        [0.1*i, 1] for i in range(21)
    ])

    Y = np.array([
        [b + m*(0.1*i + np.random.normal(mean, stddev))] for i in range(21)
    ])
    
    print('Given the following:')
    for i in range(X.shape[0]):
        print('f(' + str(X[i][0]) + ') =', Y[i][0])
        
    w = linreg_proj(X, Y)

    print('We estimate g(x) =', w[0][0], 'x +', w[1][0])
    for i in range(X.shape[0]):
        print('g(' + str(X[i][0]) + ') =', (w[0][0]*X[i][0] + w[1][0]))
    
    x = list(map(lambda x : x[0], X))
    y = list(map(lambda y : y[0], Y))
    z = list(map(lambda x : (w[0][0]*x[0] + w[1][0]), X))
    
    plt.plot(x, y, 'ro')
    plt.plot(x, z)
    
    plt.title("f(x) = " + str(w[0][0]) + "x + " + str(w[1][0]), ha='center')

    plt.show()


def logproj_demo():
     
    X = np.array(list(map(
        lambda x : [x, 1.], [
        0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50
    ])))

    Y = np.array(
        6*[[0.]] + 4*[[1.], [0.]] + 6*[[1.]]
    )
    
    print('Given the following:')
    for i in range(X.shape[0]):
        print('f(' + str(X[i][0]) + ') =', Y[i][0])
        
    w = logreg_proj(X, Y)

    print('We estimate g(x) =', w[0][0], 'x +', w[1][0])
    for i in range(X.shape[0]):
        print('g(' + str(X[i][0]) + ') =', 1.0 / (1.0 + math.exp(-w[0][0]*X[i][0] - w[1][0])))
    
    x = list(map(lambda x : x[0], X))
    y = list(map(lambda y : y[0], Y))
    z = list(map(lambda x : 1.0 / (1.0 + math.exp(-(w[0][0]*x[0] + w[1][0]))), X))
    
    plt.plot(x, y, 'ro')
    plt.plot(x, z)
    
    plt.title("$f(x) \\approx \\frac{1}{1 + e^{-(" + str(w[0][0]) + "x + " + str(w[1][0]) + "}}$").set_fontsize(20)

    ttl = plt.gca().title
    ttl.set_position([.5, 1.05])

    plt.show()
    

def logreg_demo(lr, batch, print_rate):
    
    X = np.array(list(map(
        lambda x : [x, 1.], [
        0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50
    ])))

    Y = np.array(
        6*[[0.]] + 4*[[1.], [0.]] + 6*[[1.]]
    )

    _X = X*1.0
    _Y = Y*1.0

    tmp = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(tmp)
    np.random.shuffle(Y)

    print('Given the following:')
    for i in range(X.shape[0]):
        print('f(' + str(X[i][0]) + ') =', Y[i][0])
        
    if False:
        w = logreg_sgd(X, Y, epochs=100)

        print('We estimate g(x) =', '1 / (1 + e^-(', w[0], 'x +', w[1], '))')
        for i in range(X.shape[0]):
            print('g(' + str(X[i][0]) + ') =', sigmoid(w[0]*X[i][0] + w[1]))

    
    started = False
    w = None
    while True:
        w = logreg_sgd(X, Y, epochs=print_rate, w=w, alpha = lr, batch = batch)
        x = list(map(lambda x : x[0], _X))
        y = list(map(lambda y : y[0], _Y))
        z = list(map(lambda x : 1.0 / (1.0 + math.exp(-(w[0]*x[0] + w[1]))), _X))
        
        plt.clf()
        plt.scatter(x, y, c=['r' if i == 0 else 'g' for i in y])
        plt.plot(x, z, c='k')

        plt.title("$f(x) \\approx \\frac{1}{1 + e^{-(" + str(w[0]) + "x + " + str(w[1]) + "}}$").set_fontsize(20)

        ttl = plt.gca().title
        ttl.set_position([.5, 1.05])

        plt.pause(0.1)
        

demo_name = sys.argv[1]

print('Requesting demo \'' + demo_name + '\'')

if demo_name == 'linreg' and len(sys.argv) == 6:
    linreg_demo(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
elif demo_name == 'logreg' and len(sys.argv) == 5:
    logreg_demo(float(sys.argv[2]), bool(sys.argv[3]), int(sys.argv[4]))
elif demo_name == 'logproj' and len(sys.argv) == 2:
    logproj_demo()
else:
    print("Use cases:")
    print("python", sys.argv[0], "linreg [m b mean stddev]")
    print("python", sys.argv[0], "logproj")
    print('python', sys.argv[0], "logreg [alpha batch? print_rate]")

    

