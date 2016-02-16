
#import statements

from __future__ import print_function
import os
import time
import math
import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import scale


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 30000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 400, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 25, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')

flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

NUM_CLASSES = 1


#-----------------------------------------------------------------------------------------
#Permutation and Binarization
#-----------------------------------------------------------------------------------------

def realize(X):
    def _realize_(x):
        inds = np.argsort(-(x**2).sum(axis=0)**.5+np.random.normal(0,noise,x[0].shape))
        x = x[inds,:][:,inds]*1
        x = x.flatten()[triuind]
        return x
    return np.array([_realize_(z) for z in X])


def expand(X):
    Xexp = []
    for i in range(X.shape[1]):
        for k in np.arange(0,max[i]+step,step):
            Xexp += [np.tanh((X[:,i]-k)/step)]
    return np.array(Xexp).T


def normalize(X):
    return (X-mean)/std


def transformedData(traindata):
    X = normalize(expand(realize(traindata)))
    return X


#-----------------------------------------------------------------------------------------
#Minibatch
#-----------------------------------------------------------------------------------------


class Minibatch(object):
    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.num_examples = train_data.shape[0]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.train_labels = self.train_labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return transformedData(self.train_data[start:end]), self.train_labels[start:end]




#placeholder variables to represent the input tensors



def placeholder_inputs(batch_size, trainsize):

    #input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, shape))

    #labels_placeholder = tf.placeholder(tf.float32, shape=batch_size)

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.

    train_data_node = tf.placeholder(tf.float32, shape=(batch_size, trainsize))
    train_labels_node = tf.placeholder(tf.float32, shape=(batch_size))

    #test_data_node = tf.constant(test_data)


    return train_data_node, train_labels_node


def fill_feed_dict(data, input_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }

    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.

    input_feed, labels_feed = data.next_batch(FLAGS.batch_size)

    feed_dict = {
      input_pl: input_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


def inference(input, hidden1_units, hidden2_units, dim):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      hidden1: Size of the first hidden layer.
      hidden2: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([dim, hidden1_units],
                                stddev=1.0 / math.sqrt(float(dim))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.tanh(tf.matmul(input, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2') as scope:
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.tanh(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear') as scope:
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """

    # Minimize the squared errors.
    loss = tf.reduce_mean(tf.square(logits - labels))

    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op



def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess,input_placeholder,labels_placeholder, train_set, test_set, logits, losses):
    """Runs one evaluation against the full epoch of data.

    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """

    feed_dict_train = fill_feed_dict(train_set, input_placeholder, labels_placeholder)
    logits_train = sess.run(logits, feed_dict=feed_dict_train)

    feed_dict_test = fill_feed_dict(test_set, input_placeholder, labels_placeholder)
    logits_test = sess.run(logits, feed_dict=feed_dict_test)

    return logits_train, logits_test


def run_training(train_data, train_labels, test_data, test_labels):



    train_labels_scale = scale(train_labels, axis=0)
    test_labels_scale = scale(test_labels, axis=0)

    data_sets_train = Minibatch(train_data, train_labels_scale)
    data_sets_test = Minibatch(test_data, test_labels_scale)

    getdata = transformedData(train_data)


    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        input_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size, getdata.shape[1])

        # Build a Graph that computes predictions from the inference model.

        logits = inference(input_placeholder, FLAGS.hidden1, FLAGS.hidden2, getdata.shape[1])

        # Add to the Graph the Ops for loss calculation.

        losses = loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(losses, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        #eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        n_report = train_data.shape[0] / FLAGS.batch_size
        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):


            start_time = time.time()
            # Fill a feed dictionary with the actual set of images and labels
            #  for this particular training step.

            feed_dict_train = fill_feed_dict(data_sets_train, input_placeholder, labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.

            #for i in range(229):

            #    feed_dict_train = fill_feed_dict(data_sets_train, input_placeholder, labels_placeholder)
            _, trainloss_value = sess.run([train_op, losses], feed_dict=feed_dict_train)


            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % n_report != 0:
                continue
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict_train)
            summary_writer.add_summary(summary_str, step)
            logits_train, logits_test = do_eval(sess, input_placeholder, labels_placeholder, data_sets_train, data_sets_test, logits, losses)
            # Print status to stdout.

            mae_train = np.abs((logits_train * np.std(train_labels) + np.mean(train_labels)) - train_labels).mean()
            mae_test = np.abs((logits_test * np.std(train_labels) + np.mean(train_labels)) - train_labels).mean()

            rmse_train = np.sqrt(np.square((logits_train * np.std(train_labels) + np.mean(train_labels))- train_labels).mean())
            rmse_test = np.sqrt(np.square((logits_test * np.std(train_labels) + np.mean(train_labels))- test_labels).mean())

            print('Step %d Seconds %.3f train_loss = %.2f mae_train = %.2f rmse_train = %.2f mae_test = %.2f rmse_test = %.2f' % (step, duration, trainloss_value, mae_train, rmse_train, mae_test, rmse_test))

            saver.save(sess, FLAGS.train_dir, global_step=step)



def main(_):
    run_training(X, Z, TX, TZ)




if __name__ == '__main__':

    #Prepare Datasets
    seed = 3453
    np.random.seed(seed)
    datafile = 'qm7.pkl'
    dataset = pickle.load(open(datafile, 'r'))
    split = 1

    #Train Dataset
    P = dataset['P'][range(0, split)+ range(split+1, 5)].flatten()
    X = dataset['X'][P]
    Z = dataset['T'][P]
    #Z = Z.reshape(Z.shape[0], 1)

    #Test Dataset
    Ptest = dataset['P'][split]
    TX = dataset['X'][Ptest]
    TZ = dataset['T'][Ptest]
    #TZ = TZ.reshape(TZ.shape[0], 1)

    #Initialize values
    step  = 1.0
    noise = 1.0
    triuind = (np.arange(23)[:,np.newaxis] <= np.arange(23)[np.newaxis,:]).flatten()
    max = 0
    for _ in range(10): max = np.maximum(max, realize(X).max(axis=0))
    dim_max = expand(realize(X))
    mean = dim_max.mean(axis=0)
    std = (dim_max - mean).std()


    tf.app.run()



