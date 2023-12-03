import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the embeddings from the file into a dictionary (as shown in the previous answer)
embedding_file = "embeddings.txt"
embeddings = {}
with open(embedding_file, "r", encoding="utf8") as file:
    lines = file.readlines()
    vocab_size, embedding_dim = map(int, lines[0].split())
    for line in lines[1:]:
        parts = line.split()
        word = parts[0]
        vector = np.array(list(map(float, parts[1:])))
        embeddings[word] = vector

# List of words for the 2-D projection
word_list = ["horse", "cat", "dog", "i", "he", "she", "it", "her", "his", "our", "we", "in", "on", "from", "to", "at", "by", "man", "woman", "boy", "girl", "king", "queen", "prince", "princess"]

# Extract word vectors for the specified words
word_vectors = [embeddings[word] for word in word_list]

# Apply PCA to reduce dimensionality to 2-D
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Plot the 2-D projection
plt.figure(figsize=(10, 8))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

# Annotate points with word labels
for i, word in enumerate(word_list):
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.title("2-D Projection of Word Embeddings using PCA")
plt.grid(True)
plt.show()
