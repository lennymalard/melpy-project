import melpy.NeuralNetworks as nn
from melpy.tensor import *
from melpy.preprocessing import Tokenizer

# %%
tokenizer = Tokenizer(strategy="char", lower=False)
tokenizer.load_vocabulary("inference_ressources/shakespeare_vocab")

vocab_size = len(tokenizer.value_index)

# %%
model = nn.Sequential(input_shape=(1, 1, vocab_size))

model.add(nn.Embedding(vocab_size, 128))
model.add(nn.LSTM(128, 256, activation="tanh", num_layers=1))
model.add(nn.Dense(256, vocab_size, activation="softmax"))

model.summary()

# %%
model.load_params("inference_ressources/shakespeare_parameters_01_26_2025-05_51_48.h5")

# %%
def temperature_scaling(probabilities, temperature):
    logits = np.log(probabilities)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits)

def predict_next_token(text, temperature=0.8):
    tokens = tokenizer.texts_to_sequences(text)[0]
    encoded_tokens = tokenizer.one_hot_encode(tokens).reshape(1, -1, vocab_size)
    probs = model.predict(encoded_tokens)[0]
    scaled_probs = temperature_scaling(probs, temperature)
    return np.random.choice(vocab_size, p=scaled_probs)

def generate_text(seed, length=500, context_window=50, temperature=0.8):
    generated = seed
    print(seed, end="", flush=True)
    for _ in range(length):
        context = generated[-context_window:] if len(generated) > context_window else generated
        next_token_id = predict_next_token(context, temperature)
        next_token = tokenizer.index_value[next_token_id]
        generated += next_token
        print(next_token, end="", flush=True)
    return generated

text_generated = generate_text(
    seed="\nLARTIUS:",
    length=500,
    context_window=64,
    temperature=0.6
)