import numpy as np
import random

from q1_softmax import softmax, derivative_softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
import numpy as np
from numpy import linalg


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    row_norm=linalg.norm(x,axis=1,keepdims=True)
    x=x/row_norm

    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   

    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         

    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the probabilities



    ### YOUR CODE HERE
    
    #print "predicted: ", predicted.shape
    #print "target: ",target
    #print "output Vectors: ",outputVectors.shape
    
    predicted=predicted.reshape(1,predicted.shape[0])
    scores=np.dot(predicted,outputVectors.T) #Win dot Wout gives score
    #softmax
    prob=softmax(scores).squeeze()
    cost=-np.log(prob[target])
    
    #derivative
    
    dY=prob
    dY[target]=dY[target]-1 #derivate with respect to scores
    #gradPrediction=
    dY=dY.reshape(dY.shape[0],1)
    grad_prediction=np.dot(dY.T,outputVectors) #with respect to prediction Win
    grad=np.dot(dY,predicted) #with respect to output Wout
    
    ### END YOUR CODE
    return cost,grad_prediction,grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input Specifications: same as softmaxCostAndGradient

    # Output Specifications # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    # - grad: the gradient with respect to other words


    ### YOUR CODE HERE

    # sample K tokens
    #for i in K:
    sampleIndex=[]
    for i in np.arange(0,K,1):
        negativeSamples=dataset.sampleTokenIdx()
        sampleIndex.append(negativeSamples)
    #print sampleIndex
    #print dataset
    
    
    out=np.dot(predicted,outputVectors.T)
    #out=out.reshape(out.shape[0],1)
    #print out.shape
    
    prob=sigmoid(out)
    #print prob
    #print 'a: ',np.log(1-prob[sampleIndex])
    
    cost=-1*((np.log(prob[target])) + np.sum(np.log(1-prob[sampleIndex])))
    #print cost
    
    
    #derivative
    predicted=predicted.reshape(1,predicted.shape[0])
    grad=np.zeros((outputVectors.shape[0],outputVectors.shape[1])) # VXh 5x3
    gradPred=np.zeros((predicted.shape[0],predicted.shape[1]))  # 1x3 for example for testing
    
    
    grad[target,:]=-1*(1-prob[target]) * predicted  # 1x3 for example
    
    #print gradPred.shape
    #print outputVectors[sampleIndex].shape
    #print prob[sampleIndex].shape
    #print prob
    #print 
    gradPred=(1-prob[target]) * outputVectors[target]
    predneg=-np.dot(prob[sampleIndex].reshape(prob[sampleIndex].shape[0]),outputVectors[sampleIndex])
    gradPred=(gradPred+predneg) * -1
    #print predneg.shape
    #print gradPred.shape
    #print gradPred.shape
    #print (1-prob[target]) * predicted 
    #print predicted.shape
    #print prob[target]
    
    for i in sampleIndex:
        grad[i,:]=grad[i,:]+predicted * ( prob[i])
        #print predicted * (-1 * prob[i]) #-sig(uj^T .vc) * vc
        
    
    
    ### END YOUR CODE
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    # import ipdb
    # ipdb.set_trace()
    
    #print currentWord
    #print C
    #print contextWords
    #print tokens
    #print inputVectors.shape
    #print outputVectors.shape
    
    #hotVec=np.zeros(5) #just for checking
    #hotVec[tokens[currentWord]]=1
    #hotVec=hotVec.reshape(1,hotVec.shape[0])
    #print np.dot(hotVec,inputVectors) #just for checking
    Win=inputVectors[tokens[currentWord],:] #from lookup table/matrix getting corresponding vector
    
    #skip gram at the end will give vectors of each context word, we want to 
    #maximize probability of each P(context_i|centerword)
    gradIn=np.zeros((inputVectors.shape[0],inputVectors.shape[1])) #
    gradOut=np.zeros((outputVectors.shape[0],outputVectors.shape[1]))
    
    cost = 0
    
    #print Win
    
    #over each contextword
    for i in np.arange(len(contextWords)):
        target=tokens[contextWords[i]]  #index of particular context word
        predicted=Win
        #c,gradPred,grad=softmaxCostAndGradient(predicted, target, outputVectors, dataset)
        c,gradPred,grad=word2vecCostAndGradient(predicted, target, outputVectors, dataset) # calls negsampling or softmax
        cost+=c
        #print gradPred.shape
        #print Win.shape
        #print gradIn.shape
        gradIn[tokens[currentWord]]=gradIn[tokens[currentWord]]+gradPred
        #gradIn[tokens[currentWord]]+=gradPred
        gradOut=gradOut+grad
        #print grad.shape
        #print outputVectors.shape
        
    #print word2vecCostAndGradient
    #if word2vecCostAndGradient == softmaxCostAndGradient:
        ### YOUR CODE HERE

    #    print 'Not Implemented'

        ### END YOUR CODE
    #else:
        ### YOUR CODE HERE

    #    print 'Not Implemented'

        ### END YOUR CODE

    
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################

    ### YOUR CODE HERE

    #print currentWord 
    #print contextWords
    #print inputVectors.shape #5x3
    #print outputVectors.shape #5x3
    #print C #total context words
    
    #print tokens
   
    
    sums=0
    for cw in contextWords:
        sums=sums+inputVectors[tokens[cw]]
    h=sums/(float(2*C))
    #h=h.reshape(1,h.shape[0])
    #print h
    #scores=np.dot(h,outputVectors.T) # 1xV
    #print h[tokens[currentWord]]
    c,gradPred,grad=word2vecCostAndGradient(h, tokens[currentWord], outputVectors, dataset)
    #print word2vecCostAndGradient(h, h[tokens[currentWord]], outputVectors, dataset)

    #print grad.shape
    #print gradPred.shape
    
    #print inputVectors
    gradOut=grad
    cost=c
    
    #print gradPred.shape
    gradIn=np.zeros((inputVectors.shape[0],inputVectors.shape[1]))
    for cw in contextWords:

        gradIn[tokens[cw],:]=gradIn[tokens[cw],:]+ (1/(float(2*C)) * gradPred.squeeze())
    
    
    #print gradOut
    #if word2vecCostAndGradient == softmaxCostAndGradient:
    #    print 'Not Implemented'


    #else:
    #    print 'Not Implemented'

    ### END YOUR CODE
    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset,
                                     word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], [tokens[random.randint(0, 4)] \
                                              for i in xrange(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print "\n=== Results For Skip Gram==="
    random.setstate(random.getstate())
    random.seed(31415)
    np.random.seed(9265)

    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :],
                   dataset)

    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)
    print "\n=== Results For CBOW==="
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()


#Your output should be as follows

# Testing normalizeRows...
# [[ 0.6         0.8       ]
#  [ 0.4472136   0.89442719]]
#
# ==== Gradient check for skip-gram ====
# Gradient check passed!
# Gradient check passed!
#
# ==== Gradient check for CBOW      ====
#
# === Results ===
# (11.166109001533981, array([[ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [-1.26947339, -1.36873189,  2.45158957],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ]]), array([[-0.41045956,  0.18834851,  1.43272264],
#        [ 0.38202831, -0.17530219, -1.33348241],
#        [ 0.07009355, -0.03216399, -0.24466386],
#        [ 0.09472154, -0.04346509, -0.33062865],
#        [-0.13638384,  0.06258276,  0.47605228]]))
# (array([ 13.80166427]), array([[ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ],
#        [-4.16316046, -3.85813361, -1.65076986],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ]]), array([[ 0.08029278, -0.03684413, -0.28026459],
#        [ 0.07307589, -0.0335325 , -0.25507379],
#        [-0.68292656,  0.31337605,  2.38377767],
#        [-0.73739425,  0.33836976,  2.57389893],
#        [-0.96744355,  0.443933  ,  3.37689359]]))
#
# === Results For CBOW===
# (0.79899580109066504, array([[ 0.23330542, -0.51643128, -0.8281311 ],
#        [ 0.11665271, -0.25821564, -0.41406555],
#        [ 0.11665271, -0.25821564, -0.41406555],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ]]), array([[ 0.80954933,  0.21962514, -0.54095764],
#        [-0.03556575, -0.00964874,  0.02376577],
#        [-0.13016109, -0.0353118 ,  0.08697634],
#        [-0.1650812 , -0.04478539,  0.11031068],
#        [-0.47874129, -0.1298792 ,  0.31990485]]))
# (array([ 7.14668013]), array([[-2.53925378, -3.07626957, -3.72680462],
#        [-1.26962689, -1.53813479, -1.86340231],
#        [-1.26962689, -1.53813479, -1.86340231],
#        [ 0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ]]), array([[ 0.21992784,  0.0596649 , -0.14696034],
#        [-1.37825047, -0.37390982,  0.92097553],
#        [-0.77702167, -0.21080061,  0.51922198],
#        [-3.45273868, -0.93670413,  2.30719154],
#        [-1.18374504, -0.32114184,  0.79100296]]))

