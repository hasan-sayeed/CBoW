import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scripts.utils import get_word2ix, get_files, process_data

# specifying the parameters

learning_rates = [0.01, 0.001, 0.0001]
embedding_dim = 100
batch_size = 64
max_epochs = 10
context_size = 5

# Step 1: Create a mapping between a word and an index for every word in the vocabulary

vocabulary = get_word2ix(path = "vocab.txt")   # I changed the vocab.txt file to solve the duplicated index problem the supplied file had

# Step 2: Load the data from the files and map the words to their respective index

train_data = get_files(path = "data/train")
dev_data = get_files(path = "data/dev")

processed_train_data = process_data(files = train_data, context_window = 5, word2ix = vocabulary)
processed_dev_data = process_data(files = dev_data, context_window = 5, word2ix = vocabulary)
print(type(processed_train_data[0]))

# Step 3: Creating sliding window to get the context and target words and finally adding them to create a dataset

def create_dataset(sentence_list, context_size):
    features = []
    labels = []
    for sentence in sentence_list:
        i = context_size
        while i < len(sentence) - context_size:
            window = sentence[i - context_size:i] + sentence[i + 1:i + context_size + 1]
            target = sentence[i]
            features.append(window)
            labels.append(target)
            i += 1
    print(type(features))
    return np.array(np.float64(features)), np.array(labels)

features_train, target_train = create_dataset(processed_train_data, context_size=context_size)
features_dev, target_dev = create_dataset(processed_dev_data, context_size=context_size)

# print(features_train)
print(len(features_train))
print(len(target_train))

# Step 4: Creating the datasets and dataloaders

torch.manual_seed(42)

# Create datasets using the TensorDataset class which takes in parallel tensors.
train_dataset = TensorDataset(
                          torch.tensor(np.array(features_train), dtype= torch.float),
                          torch.tensor(np.array(target_train))
                        )
dev_dataset = TensorDataset(
                          torch.tensor(np.array(features_dev), dtype= torch.float),
                          torch.tensor(np.array(target_dev))
                        )

# Once we have the datasets, we can create the data loaders.
# We choose the batch size to be 32. That means that the data will be iterated over
# in eights. We'll see this in action in the training/evaluation loop
# For the train loader, we set `shuffle=True`. This shuffles the training
# dataset before every epoch. This improves generalizability.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

print(type(train_loader))

# # Step 5: Creating the CBOW model

# class CBOW(nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super(CBOW, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.linear = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, contexts):
#         embedded_contexts = self.embeddings(contexts).sum(1)
#         return self.linear(embedded_contexts)

# # Training the model for 10 epochs



# device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# print(device)

# # Initialize some variables for keeping track of the best model
# best_model = None
# best_loss = float('inf')
# best_learning_rate = None

# # Begin the training loop

# for learning_rate in learning_rates:

#     # Initializing the model
#     model = CBOW(len(vocabulary), embedding_dim=embedding_dim).to(device)

#     # Using cross entropy loss for loss computation
#     loss_fn = nn.CrossEntropyLoss()

#     # Using Adam optimizer for optimization
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     for ep in range(1, max_epochs+1):
#         print(f"Epoch {ep}")
#         train_loss = []       
#         for inp, lab in tqdm(train_loader):
#             # Switch to train mode
#             # This doesn't make a difference to our model but is considered good practice
#             # Models might contain layers like Dropout which behave differently when training
#             # versus evaluating. This signals these layers to behave appropriately.
#             model.train()
#             optimizer.zero_grad()
#             out = model(inp.to(torch.int).to(device))    #Forward pass
#             loss = loss_fn(out, lab.to(torch.long).to(device))

#             print(f"Loss: {loss}")   # with shape {loss.shape}

#             loss.backward() # computing the gradients
#             optimizer.step()  # Performs the optimization

#             train_loss.append(loss.item())    # Appending the batch loss to the list

#         average_train_loss = np.mean(train_loss)
#         print(f"Average training batch loss for Epoch {ep}: {average_train_loss}")

#     # Check if this is the best model so far based on validation loss
#     val_loss = []
#     model.eval()  # Switch to evaluation mode
#     with torch.no_grad():
#         for inp, lab in tqdm(dev_loader):
#             out = model(inp.to(torch.int).to(device))
#             loss = loss_fn(out, lab.to(torch.long).to(device))
#             val_loss.append(loss.item())

#     average_val_loss = np.mean(val_loss)
#     print(f"Average validation batch loss for Epoch {ep}: {average_val_loss}")

#     # Check if the current model is the best based on validation loss
#     if average_val_loss < best_loss:
#         best_loss = average_val_loss
#         best_model = model.state_dict()
#         best_learning_rate = learning_rate
#         # Save the best model
#         torch.save(best_model, 'best_model.pth')

# # After the training loop completes, report the best learning rate and lowest development set loss
# print(f"Best Learning Rate: {best_learning_rate}")
# print(f"Lowest Development Set Loss: {best_loss}")

# # Load the best model
# model = CBOW(len(vocabulary), embedding_dim=embedding_dim)
# model.load_state_dict(torch.load('best_model.pth'))
# model.to(device)

# # Get the word embeddings (weights from the linear layer)
# word_embeddings = model.linear.weight
# # Move the tensor from GPU to CPU
# word_embeddings_on_cpu = word_embeddings.cpu()
# # Now you can convert it to a NumPy array
# numpy_array_of_word_embeddings = word_embeddings_on_cpu.detach().numpy()

# # Define the path for the output embeddings file
# output_file_path = 'embeddings.txt'

# # Write vocabulary size and embedding dimension to the file
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     # Write the first line with vocabulary size and embedding dimension
#     f.write(f"{len(vocabulary)} {embedding_dim}\n")

#     # Write word embeddings
#     for word_idx, embedding in enumerate(numpy_array_of_word_embeddings):
#         word = [key for key, val in vocabulary.items() if val == word_idx]
#         embedding_str = " ".join(map(str, embedding))
#         f.write(f'{word[0]} {embedding_str}\n')