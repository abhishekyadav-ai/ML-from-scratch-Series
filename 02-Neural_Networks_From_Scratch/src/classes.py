
# This file contains the complete implementation of the neural network from scratch up to the Adagrad optimizer.git 

#Importing package
import numpy as np

#Dense layer

class Layer_dense:
    #Layer intialization
    def __init__(self, n_inputs, n_neurons):
        self.weights=0.01*np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1,n_neurons))

    #Forward pass
    def forward(self, inputs):
        #remember input values
        self.inputs=inputs
        #Calculate output values from inputs, weights, biases
        self.output=np.dot(inputs, self.weights)+self.biases

    def backward(self, dvalues):
        #gradients on parameters
        self.dweights=np.dot(self.inputs.T, dvalues)
        self.dbiases=np.sum(dvalues, axis=0,keepdims=True)

        #gradient on values
        self.dinputs=np.dot(dvalues, self.weights.T)


# Relu activation 
class Activation_Relu:
    #forward pass
    def forward(self,inputs):
        #Calculate output values from input
        self.inputs=inputs
        self.output=np.maximum(0,inputs)

        #Backward pass

    def backward(self, dvalues):
        #since we need to modify the original variable 
        #Make a copy of the values first

        self.dinputs=dvalues.copy()
        self.dinputs[self.inputs <= 0] =0


#Softmax activation

class Activation_Softmax:
    # Forward Pass
    def forward(self, inputs):
        #Get unnormalized probabilities
        exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        #Normalize them for each sample
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output=probabilities

# Implementing the loss class

class Loss:
    # Calculates the data and regularization losses
    def calculate(self, output,y):
        #Calculate sample losses
        sample_losses=self.forward(output,y)
        #Calculate mean loss
        data_loss=np.mean(sample_losses)
        #return Loss
        return data_loss

#we implement the class loss just for the sake of simplicity
#So we can also see weight values

# Implemnting the categorical cross entropy class

class Loss_CategoricalCrossentropy(Loss):

    #Backward [ass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples=len(dvalues)
        #number of labels in every sample
        labels=len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape)==1:
            y_true=np.eye(labels)[y_true]
        # calculate gradient
        self.dinputs=-y_true/dvalues
        #normalize gradient
        self.dinputs=self.dinputs/samples
    #forward pass 
    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        #Clip data to prevent divisions by 0
        #Clip both sides to not drag mean towards any value
        y_pred_clipped=np.clip(y_pred, 1e-7,1-1e-7)
        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[
                range(samples),
                y_true
            ]
            #Mask values -only for one-hot encoded labels
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )
        #losses
        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods


#Combined code for softmax and Categorical cross entropy loss

class Activation_Softmax_Loss_categoricalCrossentropy:
    #creates activation and loss function objects
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Loss_CategoricalCrossentropy()

    #forward pass
    def forward(self, inputs, y_true):
        #Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output=self.activation.output
        # Calculate and return loss values
        return self.loss.calculate(self.output, y_true)
    
        #Backward pass
    def backward(self, dvalues, y_true):
        samples=len(dvalues)
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true, axis=1)
        self.dinputs=dvalues.copy()
        #calculate the gradient
        self.dinputs[range(samples), y_true]-=1
        #Normalize gradient
        self.dinputs=self.dinputs/samples

#Implementing GD with momentum and learning decay

class GD_optimizer:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_param(self, layer):
        # If we use momentum
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
           
            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


#Implementing Adagrad optimizer

class Adagrad_optimizer:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon=epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_param(self, layer):
        #if layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weights_cache'):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #update cache with squared current gradients
        layer.weights_cache+=layer.dweights**2
        layer.bias_cache +=layer.dbiases**2

        #parameter update + normalization with square rooted cache
        layer.weights+=-self.current_learning_rate*\
                 layer.dweights / \
                 (np.sqrt(layer.weights_cache)+self.epsilon) 

        layer.biases += -self.current_learning_rate * \
                layer.dbiases / \
                (np.sqrt(layer.bias_cache) + self.epsilon)     

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        