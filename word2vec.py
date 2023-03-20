import tensorflow as tf
import numpy as np
import math

sentence_cursors = None
tot_sentences = None
src_max_sent_length, tgt_max_sent_length = 0, 0
dictionary = {}
reverse_dictionary = {}
train_inputs, train_outputs = None, None
embedding_size = None 
vocabulary_size = None

def define_data_and_hyperparameters(
        _tot_sentences, 
        _src_max, 
        _tgt_max, 
        _dict, 
        rev_dict,
        _tr_inp, _tr_out, 
        _emb_size, 
        _vocab_size):
    global tot_sentences, sentence_cursors
    global src_max_sent_length, tgt_max_sent_length
    global dictionary
    global reverse_dictionary
    global train_inputs, train_outputs
    global embedding_size, vocabulary_size

    embedding_size = _emb_size
    vocabulary_size = _vocab_size
    src_max_sent_length, tgt_max_sent_length = _src_max, _tgt_max

    dictionary = _dict

    reverse_dictionary = rev_dict

    train_inputs = _tr_inp
    train_outputs = _tr_out

    tot_sentences = _tot_sentences
    sentence_cursors = [0 for _ in range(tot_sentences)]


def generate_batch_for_word2vec(batch_size, window_size, is_input):
    global sentence_cursors
    global dictionary
    global train_inputs, train_outputs
    span = 2 * window_size + 1  # [ skip_window target skip_window ]

    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    sentence_ids_for_batch = np.random.randint(0, tot_sentences, batch_size)

    for b_i in range(batch_size):
        sent_id = sentence_ids_for_batch[b_i]

        if is_input:
            buffer = train_inputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]
        else:
            buffer = train_outputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]
            
        assert buffer.size == span, 'Buffer length (%d), Current data index (%d), Span(%d)' % (
        buffer.size, sentence_cursors[sent_id], span)
        while np.all(buffer == dictionary['</s>']):
            sentence_cursors[sent_id] = 0
            sent_id = np.random.randint(0, tot_sentences)
            buffer = train_inputs[sent_id, sentence_cursors[sent_id]:sentence_cursors[sent_id] + span]
        
        batch[b_i, :window_size] = buffer[:window_size]
        batch[b_i, window_size:] = buffer[window_size + 1:]

        labels[b_i, 0] = buffer[window_size]

        if is_input:
            sentence_cursors[sent_id] = (sentence_cursors[sent_id] + 1) % (src_max_sent_length - span)
        else:
            sentence_cursors[sent_id] = (sentence_cursors[sent_id] + 1) % (tgt_max_sent_length - span)

    assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
    return batch, labels


def print_some_batches():
    global sentence_cursors, tot_sentences
    global reverse_dictionary

    sentence_cursors = [0 for _ in range(tot_sentences)]
    batch, labels = generate_batch_for_word2vec(batch_size=8, window_size=2, is_input=True)
    print('\nwith window_size = %d:' % (2))
    print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

    sentence_cursors = [0 for _ in range(tot_sentences)]

batch_size, window_size = None, None
valid_size, valid_window, valid_examples = None, None, None
num_sampled = None

train_dataset, train_labels = None, None
valid_dataset = None

softmax_weights, softmax_biases = None, None

loss, optimizer, similarity, normalized_embeddings = None, None, None, None

def define_word2vec_tensorflow(batch_size):

    global embedding_size, window_size
    global valid_size, valid_window, valid_examples
    global num_sampled
    global train_dataset, train_labels
    global valid_dataset
    global softmax_weights, softmax_biases
    global loss, optimizer, similarity
    global vocabulary_size, embedding_size
    global normalized_embeddings


    window_size = 2  
    valid_size = 20  
    valid_window = 100  

    valid_examples = np.array(np.random.randint(0, valid_window, valid_size // 2))
    valid_examples = np.append(valid_examples, np.random.randint(1000, 1000 + valid_window, valid_size // 2))
    num_sampled = 32  

    tf.compat.v1.reset_default_graph()


    train_dataset = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
    train_labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embeddings = tf.Variable(tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))
    softmax_weights = tf.Variable(tf.random.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size), dtype=tf.float32))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size], dtype=tf.float32))


    stacked_embedings = None
    print('Defining %d embedding lookups representing each word in the context' % (2 * window_size))
    for i in range(2 * window_size):
        embedding_i = tf.nn.embedding_lookup(params=embeddings, ids=train_dataset[:, i])
        x_size, y_size = embedding_i.get_shape().as_list()
        if stacked_embedings is None:
            stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
        else:
            stacked_embedings = tf.concat(axis=2,
                                          values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])

    assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
    print("Stacked embedding size: %s" % stacked_embedings.get_shape().as_list())
    mean_embeddings = tf.reduce_mean(input_tensor=stacked_embedings, axis=2, keepdims=False)
    print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())


    loss = tf.reduce_mean(
        input_tensor=tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))


    optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)


    norm = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(embeddings), axis=1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(params=normalized_embeddings, ids=valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(a=normalized_embeddings))


def run_word2vec(batch_size, num_steps = 100001):
    global embedding_size, window_size
    global valid_size, valid_window, valid_examples
    global num_sampled
    global train_dataset, train_labels
    global valid_dataset
    global softmax_weights, softmax_biases
    global loss, optimizer, similarity, normalized_embeddings
    global reverse_dictionary
    global vocabulary_size, embedding_size

    config=tf.compat.v1.ConfigProto(allow_soft_placement=True) 
    config.gpu_options.allow_growth = True	
    	
    with tf.compat.v1.Session(config=config) as session:
        tf.compat.v1.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):

            batch_data, batch_labels = generate_batch_for_word2vec(batch_size, window_size, is_input=True)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if (step + 1) % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                print('Average loss at step %d: %f' % (step + 1, average_loss))
                average_loss = 0
            if (step + 1) % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        cbow_final_embeddings = normalized_embeddings.eval()

    np.save('embeddings.npy', cbow_final_embeddings)
