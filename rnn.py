import tensorflow as tf
# tf.config.run_functions_eagerly(True)
import numpy as np
from types import SimpleNamespace
from preprocess import y_train, y_test, padded_train_sequence, padded_test_sequence, vocab_sz


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=22, embed_size=128):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()
        self.count = 0

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size  # Embed size = 128
   
        self.input_layer = tf.keras.layers.InputLayer()
       
        self.embedding = tf.keras.layers.Embedding(1000, embed_size) # Input (None, 22) -> Output (None, 22, 128)

        # new_steps = (steps - kernel_size) / stride + 1
        # kernel_size = steps - (new_steps - 1) * stride  = 22 - (18 - 1) * 1 = 5
       
        # for padding = 'same', we have output_size = input_size / stride --> stride = 3 --> pool_size = 3

        self.maxpooling = tf.keras.layers.MaxPooling1D(pool_size=3, padding='same')
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.conv1d = tf.keras.layers.Conv1D(64, 5, activation='relu')
        self.bidir_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

        # added batch normalization per paper's suggestion in the "Conclusion" section
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """
        Based upon the paper
        """
        # Note: shapes below are from the paper and don't match ours exactly

        # Input Layer takes (None, 22) -> (None, 22):
        inputs = self.input_layer(inputs)
        # Embedding Layer takes (None, 22) -> (None, 22, 128)
        inputs = self.embedding(inputs)
        # Apply SpatialDropout1D (None, 22, 128) -> (None, 22, 128):
        inputs = self.dropout(inputs)
        # Apply Conv1D to yield output shape (None, 18, 64):
        inputs = self.conv1d(inputs)
        # Add batch normalization (following convolution) in an attempt to improve original model:
        # inputs = self.batch_norm(inputs)
        # Apply MaxPooling1D to yield output shape (None, 6, 64):
        inputs = self.maxpooling(inputs)
        # Apply Bidirectional LSTM to yield output shape (None, 392):
        inputs = self.bidir_lstm(inputs)
        # Apply Dense layer to yield output shape (None, 1):
        inputs = self.dense(inputs)
    
        return inputs

    ##########################################################################################

    # Unfinished function, need to address case where word is not in vocab
    def classify_tweet(self, tweet, vocab):
        """
        Takes a model, vocab, and a tweet and returns the predicted sentiment
        """
        tweet = tweet.split()
        tweet = [vocab[word] for word in tweet]
        tweet = tf.keras.preprocessing.sequence.pad_sequences([tweet], maxlen=22, padding='post')
        logits = self.call(tweet)
        return np.argmax(logits, axis=1)[1]


#########################################################################################

def get_text_model(vocab_sz):
    
    model = MyRNN(vocab_sz)

    ## Consider changing the loss and accuracy metrics...
    loss_metric = 'binary_crossentropy'
    acc_metric  = 'accuracy'

    model.compile(
        optimizer='adam',
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )

#########################################################################################

def main():

    args = get_text_model(vocab_sz)

    args.model.fit(
        padded_train_sequence, y_train,
        epochs=5, 
        batch_size=args.batch_size,
        validation_data=(padded_test_sequence, y_test)
    )

    ## Test model on a single example:
    # example_tweet = 'I love deep learning! I\'ve learned so much in this class.'
    # args.model.classify_tweet(example_tweet, vocab)

if __name__ == '__main__':
    main()
