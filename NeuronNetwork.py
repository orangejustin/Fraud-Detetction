import tensorflow as tf

'''
A simple forward neuron network
Designed to contain only one hidden layer
Used for estimation
'''
class NeuronNetwork:
    '''
    input.shape = [sampleNumber, variableNumber, 1]
    target.shape = [sampleNumber, 1, 1]
    '''
    def __init__(self, input, target, hiddenLayerNeuronNumber=32, learningRate=0.1, bias=[0.1,0.2]):
        self.input = input
        self.target = target
        self.hiddenLayerNeuronNumber = hiddenLayerNeuronNumber
        self.learningRate = learningRate
        self.bias = bias

        #initiate parameters
        self.w1 = tf.random.normal([hiddenLayerNeuronNumber, input.shape[1]])
        self.w2 = tf.random.normal([1,hiddenLayerNeuronNumber])
        
    def setLearningRate(self, newRate):
        self.learningRate = newRate
    
    def setBias(self, newBias):
        self.bias = newBias


    def lossFunc(self, predict, target):
        temp = (predict-target)*(predict-target)
        return tf.reduce_sum(temp) / temp.shape[0]

    '''
    input.shape = [batchSize, variableNumber, 1]
    target.shape = [batchSize, 1, 1]
    '''
    def trainOneStep(self, input, target):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([self.w1, self.w2])
            hiddenOut = tf.sigmoid(tf.add(tf.matmul(self.w1, input), self.bias[0]))
            output = tf.sigmoid(tf.add(tf.matmul(self.w2, hiddenOut), self.bias[1]))
            loss = self.lossFunc(output, target)
        g1 = tape.gradient(loss, self.w1)
        g2 = tape.gradient(loss, self.w2)
        self.w1 = self.w1 - self.learningRate*g1
        self.w2 = self.w2 - self.learningRate*g2

    def calculateTotalLoss(self):
        hiddenOut = tf.sigmoid(tf.add(tf.matmul(self.w1, self.input), self.bias[0]))
        output = tf.sigmoid(tf.add(tf.matmul(self.w2, hiddenOut), self.bias[1]))
        loss = self.lossFunc(output, self.target)
        return loss

    def train(self, epoch = 128, batchSize = 128, targetLoss = 0.001):
        for i in range(epoch):
            currentPlace = 0
            averageLoss = 0
            stepCount = 0
            while True:
                if currentPlace + batchSize > self.input.shape[0]-1:
                    #will exceed the dataset
                    self.trainOneStep(self.input[currentPlace:,:,:], self.target[currentPlace:,:,:])
                    break
                else:
                    self.trainOneStep(self.input[currentPlace:currentPlace+batchSize,:,:], self.target[currentPlace:currentPlace+batchSize,:,:])
                    currentPlace += batchSize
            loss = self.calculateTotalLoss()
            print("Epoch: ", i+1, "; Loss:", end="")
            print(loss)
            if loss < targetLoss:
                return
    
    def predict(self, input):
        hiddenOut = tf.sigmoid(tf.add(tf.matmul(self.w1, input), self.bias[0]))
        output = tf.sigmoid(tf.add(tf.matmul(self.w2, hiddenOut), self.bias[1]))
        return output

