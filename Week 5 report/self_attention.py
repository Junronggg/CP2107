import numpy as np

# Define a toy vocabulary
tokens = ["The", "cat"]
vocab = ["sat", "ran", "meowed"]

# Embedding lookup (2D vectors)
embedding_dict = {
    "The": np.array([1.0, 0.0]),
    "cat": np.array([0.5, 1.0]),
}
word_embeddings = np.array([embedding_dict[t] for t in tokens])

# Positional encodings
position_encodings = np.array([[0.1, 0.2], [0.2, 0.1]])

# Step 1: Input vector = word embedding + positional encoding
input_vectors = word_embeddings + position_encodings  # shape (2, 2)

# Step 2: Weight matrices W_Q, W_K, W_V (2x2 for simplicity)
W_Q = np.array([[1, 0], [0, 1]])     # Identity
W_K = np.array([[0, 1], [1, 0]])     # Swap
W_V = np.array([[1, 1], [0, 1]])     # Mix

Q = input_vectors @ W_Q.T
K = input_vectors @ W_K.T
V = input_vectors @ W_V.T

# Step 3: Attention scores (dot product of Q and K, scaled)
dk = Q.shape[1]
scores = Q @ K.T / np.sqrt(dk)

# Step 4: Softmax attention weights
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

# Step 5: Final contextual vectors
contextual_output = attention_weights @ V

# Step 6: Predict next word using simple classifier (only for demo)
W_out = np.array([[1.0, -1.0, 0.5], [0.5, 0.5, -0.5]])  # shape (2, 3)
last_output = contextual_output[-1]  # use last token's output
logits = last_output @ W_out
probs = np.exp(logits) / np.sum(np.exp(logits))
predicted_word = vocab[np.argmax(probs)]

print("Contextual Output:\n", contextual_output)
print("Next-word Probabilities:", dict(zip(vocab, probs)))
print("Predicted Next Word:", predicted_word)
