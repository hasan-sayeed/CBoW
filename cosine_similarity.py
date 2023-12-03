import numpy as np

# Load the embeddings from the file into a dictionary
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

# Function to compute cosine similarity between two word embeddings
def cosine_similarity(word1, word2):
    if word1 in embeddings and word2 in embeddings:
        vector1 = embeddings[word1]
        vector2 = embeddings[word2]
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm1 * norm2)
        return similarity
    else:
        return None

# Example usage
word_list = ["cat", "racket"]
word1, word2 = word_list
similarity_score = cosine_similarity(word1, word2)

if similarity_score is not None:
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity_score:.2f}")
else:
    print(f"One or both of the words are not in the vocabulary.")
