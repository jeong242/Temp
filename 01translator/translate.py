import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Hyper parameters
batch_size = 128 
epochs = 200 
latent_dim = 512 
num_samples = 10000 

# Sample file names
source_file = 'sample_data/europarl-v7.et-en.en'
target_file = 'sample_data/europarl-v7.et-en.et'

#################### Data Preprocessing ####################
# Store strings from the sample data
source_texts = []
target_texts = []
# Store characters from the sample data
source_chars = []
target_chars = []

with open(source_file, 'r', encoding='utf-8') as f:
	source_lines = f.read().split('\n')
with open(target_file, 'r', encoding='utf-8') as f:
	target_lines = f.read().split('\n')

num_samples = min(num_samples, len(source_lines) - 1)

for line1, line2 in zip(source_lines[:num_samples], target_lines[:num_samples]):
	source_text =  line1.lower() 
	source_texts.append(source_text)
	for char in source_text:
		if char not in source_chars:
			source_chars.append(char)

	target_text = '😀' + line2.lower() + '🤐'
	target_texts.append(target_text)
	for char in target_text:
		if char not in target_chars:
			target_chars.append(char)
		
source_chars = sorted(source_chars)
target_chars = sorted(target_chars)
num_encoder_tokens = len(source_chars)
num_decoder_tokens = len(target_chars)
max_encoder_seq_len = max([len(txt) for txt in source_texts])
max_decoder_seq_len = max([len(txt) for txt in target_texts])

# char to index dictionary
source_dict = dict([(char, i) for i, char in enumerate(source_chars)])
target_dict = dict([(char, i) for i, char in enumerate(target_chars)])

# Store one-hot encoding of the sample data
encoder_source_data = np.zeros((num_samples, max_encoder_seq_len, num_encoder_tokens),dtype='float32')
decoder_source_data = np.zeros((num_samples, max_decoder_seq_len, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((num_samples, max_decoder_seq_len, num_decoder_tokens),dtype='float32')

for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
	# Reverse the order to increase accuracy	
	for t, char in enumerate(source_text):
		encoder_source_data[i, len(source_text)-t-1, source_dict[char]] = 1.
	for t, char in enumerate(target_text):
		# decoder_target_data is ahead of decoder_source_data by one timestep
		decoder_source_data[i, t, target_dict[char]] = 1.
		if t > 0:
			decoder_target_data[i, t - 1, target_dict[char]] = 1.

#################### Model Structuring ####################
### Build Encoder
# Encoder Input Layer
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Encoder LSTM Layer 1
encoder1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoded, h1, c1 = encoder1(encoder_inputs)
encoder_states1 = [h1, c1]
# Encoder LSTM Layer 2
encoder2 = LSTM(latent_dim, return_state=True)
_, h2, c2 = encoder2(encoded, initial_state=encoder_states1)
encoder_states2 = [h2, c2]

### Build Decoder
# Decoder Input Layer
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# Decoder LSTM Layer 1
decoder1 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoded, h3, c3 = decoder1(decoder_inputs, initial_state=encoder_states2)
decoder_states = [h3, c3]
# Decoder LSTM Layer 2
decoder2 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoded, _, _ = decoder2(decoded, initial_state=decoder_states)
# Decoder Dense Layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoded = decoder_dense(decoded)

### Build the overall Encoder-Decoder Model
model = Model([encoder_inputs, decoder_inputs], decoded)

#################### Model Training ####################
# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_source_data, decoder_source_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
	  verbose=1,
          validation_split=0.2)

print('Saving model')
model.save('translator.h5')

#################### Prediction ####################
# Define prediction models
encoder_model = Model(encoder_inputs, encoder_states2)

decoder_state_source_h = Input(shape=(latent_dim,))
decoder_state_source_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_source_h, decoder_state_source_c]
decoder_outputs, state_h, state_c = decoder1(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs, state_h, state_c = decoder2(decoder_outputs, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_states = [state_h, state_c]
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# index to char dictionary
reverse_source_dict = dict((i, char) for char, i in source_dict.items())
reverse_target_dict = dict((i, char) for char, i in target_dict.items())

def one_hot_vectorization(sentence):
	seq = np.zeros((1,len(sentence),num_encoder_tokens),dtype='float32')
	for i, char in enumerate(sentence.lower()):
		seq[0,i,source_dict[char]] = 1
	
	return seq

# Input should be numpy array with shape == (None, max_length_sentence, num_encoder_tokens) 
def predict_from_batch(source_seq):	
	
	states_value = encoder_model.predict(source_seq)
	
	target_seq = np.zeros((1, 1, num_decoder_tokens))

	target_seq[0, 0, target_dict['😀']] = 1.
		
	decoded_sentence = ''
	while True:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
		
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_dict[sampled_token_index]
		decoded_sentence += sampled_char
		
		if (sampled_char == '🤐' or len(decoded_sentence) > max_decoder_seq_len):
			break
			
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.
		
		states_value = [h, c]
		
	return decoded_sentence

# Input should be string with every character exists in the source_chars 
def predict(sentence):
	for char in sentence.lower():
		if char not in source_chars:
			print('Invalid character: '+char)
			return None

	print(predict_from_batch( one_hot_vectorization(sentence) ))

