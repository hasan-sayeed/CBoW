import numpy as np

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

# Function to find the most similar word to a given vector using cosine similarity
def find_most_similar_word(vector, exclude_words=[]):
    max_similarity = -1
    most_similar_word = None

    for word, word_vector in embeddings.items():
        if word not in exclude_words:
            similarity = np.dot(vector, word_vector) / (np.linalg.norm(vector) * np.linalg.norm(word_vector))
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_word = word

    return most_similar_word

# Define the analogy
word_a = "woman"
word_b = "man"
word_c = "princess"

# Calculate the vector for the analogy completion
vector_a = embeddings[word_a]
vector_b = embeddings[word_b]
vector_c = embeddings[word_c]
analogy_completion_vector = vector_b - vector_a + vector_c

# Find the most similar word to the analogy_completion_vector
most_similar_word = find_most_similar_word(analogy_completion_vector, exclude_words=[word_a, word_b, word_c])

print(f"{word_a}:{word_b} :: {word_c}:{most_similar_word}")
