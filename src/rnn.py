import numpy as np
import tensorflow as tf

"""
RNN
no batching
loss is based on last word in sequence only
"""

'''
ssh superra@138.23.191.212
scp rnn.py superra@138.23.191.212:/home/superra/learn_neural_nets/annie/rnn.py
'''

# hyper-parameters
NUM_HIDDENS = 3  # need at least 2
LEARNING_RATE = 0.1
NUM_EPOCHS = 100
NO_CONTEXT = False


def one_hot(v):
    return np.eye(num_types)[v]


# data
sequences = np.asarray([
    ['A1', 'X1', 'B1'],
    ['A1', 'X2', 'B1'],
    ['A2', 'X1', 'B2'],
    ['A2', 'X2', 'B2']])
num_steps = sequences.shape[1] - 1 if not NO_CONTEXT else 1
types = np.unique(sequences)
num_tokens, num_types = len(sequences), len(types)
word_to_ix = {word: ix for ix, word in enumerate(types)}
ix_to_word = {i: word for i, word in enumerate(types)}
xs = []
ys = []
for sequence in sequences:
    x = one_hot([word_to_ix[word] for word in sequence[:-1]])
    y = one_hot([word_to_ix[sequence[-1]]])
    if NO_CONTEXT:
        x = x[-1, :][:, np.newaxis].T
        print(x)
    xs.append(x)
    ys.append(y)

# graph
tf_x = tf.placeholder(shape=[None, num_types], dtype=tf.float32, name="inputs")
tf_y = tf.placeholder(shape=[None, num_types], dtype=tf.float32, name="targets")

# sets the initial settings because there is no layer before the first layer
init_state = tf.constant(0, shape=[1, NUM_HIDDENS], dtype=tf.float32, name="state")
initializer = tf.random_normal_initializer(stddev=0.1)
with tf.variable_scope("RNN") as scope:
    h_t = init_state
    logits = []
    for t, x_t in enumerate(tf.split(tf_x, num_steps, axis=0)):
        if t > 0:
            scope.reuse_variables()  # variables are created once and then reused
        Wxh = tf.get_variable("Wxh", [num_types, NUM_HIDDENS], initializer=initializer)
        Whh = tf.get_variable("Whh", [NUM_HIDDENS, NUM_HIDDENS], initializer=initializer)
        Why = tf.get_variable("Why", [NUM_HIDDENS, num_types], initializer=initializer)
        bh = tf.get_variable("bh", [NUM_HIDDENS], initializer=initializer)
        by = tf.get_variable("by", [num_types], initializer=initializer)
        h_t = tf.tanh(tf.matmul(x_t, Wxh) + tf.matmul(h_t, Whh) + bh)
        y_t = tf.matmul(h_t, Why) + by
        logits.append(y_t)
tf_last_logit = logits[-1]
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=tf_last_logit))
update_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(tf_loss)

# session
'''
- protects variables from changing for each session
- interface for the graph
- update_step is where the actual training is happening (gradient descent)
- tf_loss is the actual value from -log(actual - prediction)
'''
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# train loop
for epoch_id in range(NUM_EPOCHS):
    for x, y in zip(xs, ys):
        loss, _ = sess.run([tf_loss, update_step], feed_dict={tf_x: x, tf_y: y})
        print('epoch={} loss={}'.format(epoch_id, loss))

# accuracy
'''
- gets prediction of which one should be next
- if the prediction is the answer, add 1 to the number of correct
'''
num_correct = 0
num_total = 0
for x, y in zip(xs, ys):
    last_logit = sess.run(tf_last_logit, feed_dict={tf_x: x})
    prediction = ix_to_word[np.argmax(last_logit)]
    answer = ix_to_word[np.argmax(y)]
    print(prediction, answer)
    if prediction == answer:
        num_correct += 1
    num_total += 1
accuracy = num_correct / num_total
print('accuracy={}'.format(accuracy))
