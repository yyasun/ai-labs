import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from anfis import ANFIS


# Generate Xset
D = 4  # number of regressors
T = 1  # delay
N = 2000  # Number of points to generate
X = np.loadtxt("trainset.txt", usecols=[0,1,2])
Y = np.loadtxt("trainset.txt", usecols=[3])

trnX = X[:Y.size - round(Y.size * 0.3), :]
trnY = Y[:Y.size - round(Y.size * 0.3)]
chkX = X[Y.size - round(Y.size * 0.3):, :]
chkY = Y[Y.size - round(Y.size * 0.3):]

# ANFIS params and Tensorflow graph initialization
m = 16  # number of rules
alpha = 0.01  # learning rate

fis = ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)

# Training
num_epochs = 10

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    # Initialize model parameters
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        #  Run an update step
        trn_loss, trn_pred = fis.train(sess, trnX, trnY)
        # Evaluate on validation set
        val_pred, val_loss = fis.infer(sess, chkX, chkY)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)
    # Plot the cost over epochs
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(np.squeeze(trn_costs))
    plt.title("Training loss, Learning rate =" + str(alpha))
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(val_costs))
    plt.title("Validation loss, Learning rate =" + str(alpha))
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    # Plot resulting membership functions
    fis.plotmfs(sess)
    plt.show()