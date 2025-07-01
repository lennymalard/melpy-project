import melpy.NeuralNetworks as nn
from melpy.Tensor import *
from melpy.preprocessing import Tokenizer

# %% Tokenizer loading
tokenizer = Tokenizer(strategy="char", lower=False)
tokenizer.load_vocabulary("inference_ressources/full_shakespeare_vocab")

vocab_size = len(tokenizer.value_index)

# %% Model loading
model = nn.Sequential(input_shape=(1, 1, vocab_size))

model.add(nn.Embedding(vocab_size, 128))
model.add(nn.LSTM(128, 256, activation="tanh", num_layers=2))
model.add(nn.Dense(256, vocab_size, activation="softmax"))

model.summary()

model.load_params("inference_ressources/shakespeare_parameters_01_31_2025-20_52_29.h5")

# %% Inference functions definition
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

def generate_text(seed, temperature=0.8, length=500, context_window=64):
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
    seed="\nBRUTUS:",
    length=1000,
    context_window=32,
    temperature=0.6
)

"""# %% Gradio app creation
# To use it, just click on the local URL given in the Python console.
import gradio as gr

app_inputs =  [
    gr.Textbox(label="Seed", value="BRUTUS:"),
    gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.1, value=0.8),
    gr.Slider(label="Text Length", minimum=50, maximum=5000, step=50, value=500)
]

app = gr.Interface(fn=generate_text, inputs=app_inputs, outputs="textbox")

if __name__ == "__main__":
    app.launch()"""
