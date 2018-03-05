#MNIST Dataset CNN
'''
REPEAT:
input > weight > hidden layer 1 (activation function) > weights > hiden l2
(activation function) > weights > output Layer

compare output to indented output > using cost or loss function (exp: cross entropy)
optimization function (optimizer) > minimize the cost (exp: Adamoptimizer, SGD, Adagrad)

backpropagation

feed forward + backprop = epoch (x times)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Download MNIST dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128 #get 100 images as input and do feed forward at a times

# height X width (28*28 = 784)
x = tf.placeholder('float',[None,784]) #input data => format hard-coded
y = tf.placeholder('float') #output data

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#extract features
#one pixel at a time
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

#simplify features	
#pooling 2x2 at a time
def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	
	
def convolutional_neural_network(x):

    #parameter => 5x5 matrix; 1 input; 32 output features;
	#fully connected layer parameter => 7X7 matrix (mul) 64 features
	weights = { 'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
				'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
				'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
				'out':tf.Variable(tf.random_normal([1024,n_classes]))}
			  
    #biases is for just number of outputs			  
	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			  'b_conv2':tf.Variable(tf.random_normal([64])),
			  'b_fc':tf.Variable(tf.random_normal([1024])),
			  'out':tf.Variable(tf.random_normal([n_classes]))}
	
	#reshape to new shape
	x = tf.reshape(x, shape=[-1, 28, 28, 1])
	
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)
	
	fc = tf.reshape(conv2, [-1,7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)
	
	output = tf.matmul(fc, weights['out']) + biases['out']	
	
	return output
	
def train_neural_network(x):
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
	
	#parameter : Rate = 0.001 (default)
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	#cycles = feed forward  + backprop
	hm_epochs = 10
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size) #gotta build this function for other examples
				_,c = sess.run([optimizer, cost], feed_dict = {x: epoch_x,y: epoch_y})
				epoch_loss += c
				
			print('Epoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		

train_neural_network(x)		
print("DONE...")




