import tensorflow as tf
import pathlib

cache_dir = './tmp'
dataset_file_name = 'shakespeare.txt'
dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

dataset_file_path = tf.keras.utils.get_file(
    fname=dataset_file_name,
    origin=dataset_file_origin,
    cache_dir=pathlib.Path(cache_dir).absolute()
)

# %%
from melpy.preprocessing import Tokenizer, generate_sequence_dataset

# Reading the database file.
text = open(dataset_file_path, mode='r').read()

print('Length of text: {} characters'.format(len(text)))

print(text[:250])

tokenizer = Tokenizer(strategy="character", lower=False)
tokens = tokenizer.texts_to_sequences(text)[0]

X_train, y_train = generate_sequence_dataset(tokens, 16)

vocab_size = len(tokenizer.value_index)
batch_size = X_train.shape[0]

X_train = tokenizer.one_hot_encode(X_train)
y_train = tokenizer.one_hot_encode(y_train)

X_train = X_train.reshape(batch_size,-1, vocab_size)

tokenizer.save_vocabulary("full_shakespeare_vocab")

# %%
import melpy.NeuralNetworks as nn
from melpy.tensor import *

model = nn.Sequential(X_train, y_train)

model.add(nn.Embedding(X_train.shape[-1], 128))
model.add(nn.LSTM(128, 256, activation="tanh", num_layers=2))
model.add(nn.Dense(256, y_train.shape[-1], activation="softmax"))

model.compile(nn.CategoricalCrossEntropy(), nn.Adam(learning_rate= 0.01))
model.summary()

# %%
model.fit(epochs=10, batch_size=256, verbose=2)
model.results()

model.save_params("shakespeare_parameters")
model.save_histories("shakespeare_history")

# %%
def temperature_scaling(probabilities, temperature):
    logits = np.log(probabilities)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits)

def predict_next_token(text, temperature=1.0):
    tokens = tokenizer.texts_to_sequences(text)[0]
    encoded_tokens = tokenizer.one_hot_encode(tokens).reshape(1, -1, vocab_size)
    probs = model.predict(encoded_tokens)[0]
    scaled_probs = temperature_scaling(probs, temperature)
    return np.random.choice(vocab_size, p=scaled_probs)

def generate_text(seed, length=500, context_window=50, temperature=0.8):
    generated = seed
    for _ in range(length):
        context = generated[-context_window:] if len(generated) > context_window else generated
        next_token_id = predict_next_token(context, temperature)
        generated += tokenizer.index_value[next_token_id]
    return generated

text_generated = generate_text(
    seed="LARTIUS: ",
    length=500,
    context_window=64,
    temperature=0.7
)

print(text_generated)