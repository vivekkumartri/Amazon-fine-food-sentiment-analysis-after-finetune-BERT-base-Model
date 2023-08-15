import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES']='1'

df=pd.read_csv('/home/vivek.trivedi/Reviews.csv',sep=",")
reviews=df['Text'].to_numpy()
def mark_sentiment(rating):
    if(rating<3):
        return 0 # negative 
    else:
        return 1 # positive
labels=df['Score'].apply(mark_sentiment).to_numpy()
print(reviews[:2000])
print(labels[:20])

from string import punctuation

print(punctuation)

all_text = '\n'.join(reviews)

# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

words[:30]

# feel free to use this import
from collections import Counter

## Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab,1)}

## use the dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
  reviews_ints.append([vocab_to_int[word] for word in review.split()])

# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))  # should ~ 74000+
print()

# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])

encoded_labels = labels

# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

print('Number of reviews before removing outliers: ', len(reviews_ints))

## remove any reviews/labels with zero length from the reviews_ints list.

## get any indices of any reviews with length 0
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# remove 0-length review with their labels
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_ints))

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''
    ## getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    ## for each review, I grab that review
    for i, row in enumerate(reviews_ints):
      features[i, -len(row):] = np.array(row)[:seq_length]

    return features

# Test your implementation!

seq_length = int(np.mean(list(review_lens.keys())))

features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30,:10])

split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)
split_idx = int(len(features)*0.8)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x))
test_y,val_y = remaining_y[:test_idx], remaining_y[test_idx:]
test_x,val_x = remaining_x[:test_idx], remaining_x[test_idx:]


## print out the shapes of your resultant feature data
print("\t\t\tFeatures Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 20

# make sure to SHUFFLE your data

_ = torch.manual_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
output_size = 1
embedding_dim = 300
hidden_dim = 256
n_layers = 2
n_epoch=10

class MyRNN(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(MyRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size,1)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        hidden_state = self.init_hidden(batch_size).to(device)
        output, hidden_state = self.rnn(embeds,hidden_state)
        output = self.fc(hidden_state.squeeze())
        output=self.sig(output)
        #output = output.view(batch_size, -1)
        return output[-1]
    def init_hidden(self,batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

def accuracy_loss(model,dataset,criterion):
  num_correct = 0
  num_samples = len(dataset)*batch_size
  model.eval()
  loss_=0
  with torch.no_grad():
      for name, label in dataset:
          output = model(name.to(device))
          loss = criterion(output.float(), label.view(-1,1).to(device).float())
          pred = torch.round(output.squeeze())
          num_correct += sum(pred == label.to(device)).cpu().numpy()
          loss_+=loss.item()
  return (num_correct / num_samples,loss_/num_samples)

hiden_size_list=[64*i for i in range(1,6)]
learning_rate_list=[1e-5,1e-4,1e-3,1e-2]
# accuracy_list={}
# for learning_rate in tqdm(learning_rate_list):
#   accuracy_list[learning_rate]={}
#   for hidden_size in tqdm(hiden_size_list):
#     model = MyRNN(2, hidden_size).to(device)
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     for epoch in range(n_epoch):
#         acc_epoch=[]
#         model.train().to(device)
#         train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#         #valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
#         test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
#         for  name, label in train_loader:
#             model.zero_grad()
#             output = model(name.to(device))
#             loss = criterion(output.float(), label.view(-1,1).to(device).float())
#             loss.backward()
#             optimizer.step()
#         acc_epoch.append([accuracy_loss(model,train_loader,criterion),accuracy_loss(model,test_loader,criterion)])
#         print('learning rate =',learning_rate,'hidden size =',hidden_size,'epoch =',epoch,'\n train accuracy,train loss,test accuracy,test loss',acc_epoch[-1])
#     accuracy_list[learning_rate][hidden_size]=acc_epoch
#     with open("/home/vivek.trivedi/accuracy_loss_list_RNN.pkl",'wb') as file:
#         pickle.dump(accuracy_list,file)

class MyGRU(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(MyGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size,1)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        hidden_state = self.init_hidden(batch_size).to(device)
        output, hidden_state = self.gru(embeds,hidden_state)
        output = self.fc(hidden_state.squeeze())
        output=self.sig(output)
        #output = output.view(batch_size, -1)
        return output[-1]
    def init_hidden(self,batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

hiden_size_list=[64*i for i in range(1,6)]
learning_rate_list=[1e-5,1e-4,1e-3,1e-2]
# accuracy_list={}
# for learning_rate in tqdm(learning_rate_list):
#   accuracy_list[learning_rate]={}
#   for hidden_size in tqdm(hiden_size_list):
#     model = MyGRU(2, hidden_size).to(device)
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     acc_epoch=[]
#     for epoch in range(n_epoch):
#         model.train().to(device)
#         train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
#         #valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
#         test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
#         for  name, label in tqdm(train_loader):
#             model.zero_grad()
#             output = model(name.to(device))
#             loss = criterion(output.float(), label.view(-1,1).to(device).float())
#             loss.backward()
#             optimizer.step()
#         acc_epoch.append([accuracy_loss(model,train_loader,criterion),accuracy_loss(model,test_loader,criterion)])
#         print('learning rate =',learning_rate,'hidden size =',hidden_size,'epoch =',epoch,'\n train accuracy,train loss,test accuracy,test loss',acc_epoch[-1])
#     accuracy_list[learning_rate][hidden_size]=acc_epoch
#     with open("/home/vivek.trivedi/accuracy_loss_list_gru.pkl",'wb') as file:
#       pickle.dump(accuracy_list,file)

import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
       
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden

def accuracy_loss(net,loader):
    losses = [] # track loss
    num_correct = 0

    # init hidden state


    net.eval()
    # iterate over test data
    for inputs, labels in loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history

        inputs, labels = inputs.to(device), labels.to(device)

        # get predicted outputs
        output, h = net(inputs)

        # calculate loss
        loss = criterion(output.squeeze(), labels.float())
        losses.append(loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    np.mean(losses)
    acc = num_correct/len(loader.dataset)
    return acc,np.mean(losses)

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1 # +1 for zero padding + our word tokens
output_size = 1
embedding_dim = 400
n_layers = 2
accuracy_list={}
for lr in learning_rate_list:
    accuracy_list[lr]={}
    for hidden_dim in hiden_size_list:
        net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        counter = 0
        print_every = 100
        clip=5 # gradient clipping
        acc_epoch=[]
        for e in range(n_epoch):
            train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
            #valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
            test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
            # initialize hidden state
            
            # batch loop
            net.train()
            for inputs, labels in tqdm(train_loader):
                counter += 1

                inputs, labels = inputs.to(device), labels.to(device)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history

                # zero accumulated gradients
                net.zero_grad()

                # get the output from the model
                output, h = net(inputs)

                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()
            acc_epoch.append([accuracy_loss(net,train_loader),accuracy_loss(net,test_loader)])
            print('learning rate =',lr,'hidden size =',hidden_dim,'epoch =',e,'\n train accuracy,train loss,test accuracy,test loss',acc_epoch[-1])
        accuracy_list[lr][hidden_dim]=acc_epoch
        with open("/home/vivek.trivedi/accuracy_loss_list_lstm.pkl",'wb') as file:
            pickle.dump(accuracy_list,file)

