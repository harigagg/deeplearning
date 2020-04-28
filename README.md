# Neural network for classification

ann.py: This project makes use of churn_modelling.csv file to determine if a customer stays in the bank or leaves the bank based on various features like creditscore, tenure, gender, geography, salary, etc.. An Artificial neural network using Keras was built for predicting if a customer stays or leaves the bank with an accuracy of 86% on train and test

## Guide to build a NN:

STEP1: Randomly Initialize the weights to small numbers close to 0 (make sure its !=0, why???A: Causes Symmetries in the model. All the hidden activations will be identical, leading to the weights feeding into a given hidden unit will have identical derivatives. Therefore, these weights will have identical values in the next step, and so on. With nothing to distinguish different hidden units, no learning will occur. This phenomenon is perhaps the most important example of a saddle point in neural net training)
STEP2: Input the first observation of your dataset in the input layers each feature in one input node.
STEP3: Forward-propagation: from left to right, the neurons are activated in a way that the impact of each neuron’s activation is limited by the weights. Propagate the activations until getting the predicted result y.
STEP4: Compare the predicted result with the actual result. Measure the generated error.
STEP5: Back-propagation: from right to left, the error is backpropagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.
STEP6: Repeat steps 1 to 5 and update the weights after each observation (Reinforcement learning) or Repeat 1 to 5 and update the weights after a batch of observations(batch learning).
STEP7: When the whole training set passed through the ANN, that makes an epoch. Redo for more epochs

