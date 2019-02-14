import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

emb_file = '/home/grtzsohalf/Desktop/LibriSpeech/phonetic_feature_txt_average_250'
log_dir = '/home/grtzsohalf/Desktop/LibriSpeech/phonetic_feature_txt_250_log'
# emb_file = '/home/grtzsohalf/Desktop/LibriSpeech/phonetic_all_average_250'
# log_dir = '/home/grtzsohalf/Desktop/LibriSpeech/phonetic_all_250_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

test_words = ['OTHER', 'OTHERS', 'TIME', 'TIMES', 'EYE', 'EYES']

embedding = []
words = []
count = 0
with open(emb_file, 'r') as f_emb:
    f_emb.readline()
    for line in tqdm(f_emb):
        line = line.strip().split()
        word = line[0]
        if not word in test_words:
            continue
        feat = list(map(float, line[1:]))
        embedding.append(feat)
        words.append(word)
        count += 1
embedding = np.array(embedding)

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
   for word in words:
        f.write(word + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save model
saver = tf.train.Saver()
saver.save(sess, os.path.join(log_dir, "model.ckpt"))

