import helper
import re

#載入數據庫
dir = './data/ce.txt'
text = helper.load_text(dir)
test=str(text[:100])

#列印字數及首100字
print("資料庫字數： "+str(len(text)))
print("首一百字內容為： "+test)

#只用首10萬字作訓練
num_words_for_training = 100000
text = text[:num_words_for_training]

#分段
lines_of_text = text.split('\n')
print("有多少段落： "+str(len(lines_of_text)))
print("首15段為： ")
print(lines_of_text[:15])

#去空行
lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]
#清除每段首尾空格
lines_of_text = [lines.strip() for lines in lines_of_text]

#清除全形空格 \u3000
lines_of_text = [lines.replace(u'\u3000', u'') for lines in lines_of_text]


#將文字對應到數字

def create_lookup_tables(input_data):
    
    vocab = set(input_data)
    
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab

#創建一個符號查詢表，把逗號，句號等符號與一個標誌一一對應，用於將『我。 』和『我』這樣的類似情況區分開來，排除標點符號的影響。

def token_lookup():

    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])
    
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]

    return dict(zip(symbols, tokens))

#預處理一下數據，並保存到磁盤
helper.preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)

#讀取我們需要的數據
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

#檢查改一下當前Tensorflow的版本以及是否有GPU可以使用

import problem_unittests as tests
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

'''
正式進入創建RNN的階段了。

我們的RNN不是原始RNN了，中間一定要使用到LSTM和word2vec的功能。下面將基於Tensorflow，創建一個帶2層LSTM層的RNN網絡來進行訓練。

首先設置一下超參。
'''

# 訓練循環次數
num_epochs = 5

# batch大小 256
batch_size = 128

# lstm層中包含的unit個數 512
rnn_size = 256

# embedding layer的大小 512
embed_dim = 256

# 訓練步長 30
seq_length = 30

# 學習率 0.003
learning_rate = 0.003

# 每多少步打印一次訓練信息 30
show_every_n_batches = 30

# 保存session狀態的位置
save_dir = './save'

def get_inputs():
    
    # inputs和targets的類型都是整數的
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
 	# lstm層數
 	num_layers = 3
 	# dropout時的保留概率
 	keep_prob = 0.8
 	# 創建包含rnn_size個神經元的lstm cell
 	cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
 	# 使用dropout機制防止overfitting等
 	drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
 	# 創建2層lstm層
 	cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])
 	# 初始化狀態為0.0
 	init_state = cell.zero_state(batch_size, tf.float32)
 	# 使用tf.identify給init_state取個名字，後面生成文字的時候，要使用這個名字來找到緩存的state
 	init_state = tf.identity(init_state, name='init_state')

 	return cell, init_state

def get_embed(input_data, vocab_size, embed_dim):

	# 先根據文字數量和embedding layer的size創建tensorflow variable
	embedding = tf.Variable(tf.truncated_normal([vocab_size, embed_dim], stddev=0.1), dtype=tf.float32, name="embedding")
	# 讓tensorflow幫我們創建lookup table
	return tf.nn.embedding_lookup(embedding, input_data, name="embed_data")

#創建rnn節點，使用dynamic_rnn方法計算出output和final_state

def build_rnn(cell, inputs):
	'''
	cell就是上面get_init_cell創建的cell
	'''
	outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
	# 同樣給final_state一個名字，後面要重新獲取緩存
	final_state = tf.identity(final_state, name="final_state")

	return outputs, final_state

#用上面定義的方法創建rnn網絡，並接入最後一層fully_connected layer計算rnn的logits

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
	# 創建embedding layer
	embed = get_embed(input_data, vocab_size, embed_dim)
	# 計算outputs 和 final_state
	outputs, final_state = build_rnn(cell, embed)
	# remember to initialize weights and biases, or the loss will stuck at a very high point
	logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None, weights_initializer = tf.truncated_normal_initializer(stddev=0.1), biases_initializer=tf.zeros_initializer())
# logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
	return logits, final_state   

#那麼大的數據量不可能一次性都塞到模型裡訓練，所以用get_batches方法一次使用一部分數據來訓練

def get_batches(int_text, batch_size, seq_length):
	# 計算有多少個batch可以創建
	# n_batches = (len(int_text) // (batch_size * seq_length))
	# 計算每一步的原始數據，和位移一位之後的數據
	# batch_origin = np.array(int_text[: n_batches * batch_size * seq_length])
	# batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

	# 將位移之後的數據的最後一位，設置成原始數據的第一位，相當於在做循環
	# batch_shifted[-1] = batch_origin[0]

	# batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)
	# batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)

	# batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))

	characters_per_batch = batch_size * seq_length
	num_batches = len(int_text) // characters_per_batch

	# clip arrays to ensure we have complete batches for inputs, targets same but moved one unit over
	input_data = np.array(int_text[ : num_batches * characters_per_batch])
	target_data = np.array(int_text[1 : num_batches * characters_per_batch + 1])

	inputs = input_data.reshape(batch_size, -1)
	targets = target_data.reshape(batch_size, -1)

	inputs = np.split(inputs, num_batches, 1)
	targets = np.split(targets, num_batches, 1)

	batches = np.array(list(zip(inputs, targets)))
	batches [-1][-1][-1][-1] = batches [0][0][0][0]

	return batches

#創建整個RNN網絡模型
# 導入seq2seq，下面會用他計算loss

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()

with train_graph.as_default():
	# 文字總量
	vocab_size = len(int_to_vocab)
	# 獲取模型的輸入，目標​​以及學習率節點，這些都是tf的placeholder
	input_text, targets, lr = get_inputs()
	# 輸入數據的shape
	input_data_shape = tf.shape(input_text)

	# 創建rnn的cell和初始狀態節點，rnn的cell已經包含了lstm，dropout
	# 這裡的rnn_size表示每個lstm cell中包含了多少的神經元
	cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
	# 創建計算loss和finalstate的節點
	logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
	# 使用softmax計算最後的預測概率
	probs = tf.nn.softmax(logits, name='probs')

	# 計算loss
	cost = seq2seq.sequence_loss(
		logits,
		targets,
		tf.ones([input_data_shape[0], input_data_shape[1]]))

	# 使用Adam提督下降
	optimizer = tf.train.AdamOptimizer(lr)

	# 裁剪一下Gradient輸出，最後的gradient都在[-1, 1]的範圍內
	gradients = optimizer.compute_gradients(cost)
	capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
	train_op = optimizer.apply_gradients(capped_gradients)

#開始訓練模型
# 獲得訓練用的所有batch
batches = get_batches(int_text, batch_size, seq_length)

# 打開session開始訓練，將上面創建的graph對像傳遞給session
with tf.Session(graph=train_graph) as sess:
	sess.run(tf.global_variables_initializer())

	for epoch_i in range(num_epochs):
		state = sess.run(initial_state, {input_text: batches[0][0]})

		for batch_i, (x, y) in enumerate(batches):
			feed = {
				input_text: x,targets: y, initial_state: state, lr: learning_rate}
			train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

			# 打印訓練信息
			if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
				print('Epoch {:>3} Batch {:>4}/{} train_loss = {:.3f}'.format(
					epoch_i,
					batch_i,
					len(batches),
					train_loss))

	# 保存模型
	saver = tf.train.Saver()
	saver.save(sess, save_dir)
	print('Model Trained and Saved')