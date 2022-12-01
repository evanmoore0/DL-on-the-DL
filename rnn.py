import tensorflow as tf
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

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size # RNN size = 22 from paper (I think)
        self.embed_size = embed_size  # Embed size = 128 from paper

        ## Define an embedding component to embed the word indices into a trainable embedding space.
        ## Define a recurrent component to reason with the sequence of data. 
        ## You may also want a dense layer near the end...    
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size) # Input (None, 22) -> Output (None, 22, 128)


    def call(self, inputs):
        """
        Based entirely upon the paper
        """
        inputs = self.embedding(inputs)
        # Apply SpatialDropout1D:
        inputs = tf.keras.layers.SpatialDropout1D(0.1)(inputs) # TODO: Experiment w/ this parameter
        # Apply Conv1D to yield output shape (None, 18, 64):
        inputs = tf.keras.layers.Conv1D(64, 5, activation='relu')(inputs)
        # Apply MaxPooling1D to yield output shape (None, 6, 64):
        inputs = tf.keras.layers.MaxPooling1D(3)(inputs) # pool_size=3 (?)
        # Apply Bidirectional LSTM to yield output shape (None, 392):
        inputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(196, return_sequences=True))(inputs)
        # Apply Dense layer to yield output shape (None, 1):
        inputs = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)
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

    class PerplexityMetric(tf.keras.metrics.Metric):
        def __init__(self, name='perplexity', **kwargs):
            super().__init__(name=name, **kwargs)
            self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            self.perplexity = self.add_weight(name='self perplexity', initializer='zeros')
        def update_state(self, y_true, y_pred, sample_weight=None):
            loss = self.cross_entropy(y_true, y_pred)
            self.perplexity = tf.exp(tf.reduce_mean(loss))
        def result(self):
            return self.perplexity
        def reset_states(self):
            self.perplexity = 0.0

    ## Consider changing the loss and accuracy metrics...
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric  = PerplexityMetric()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
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
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(padded_test_sequence, y_test)
    )

    ## Test model on a single example:
    # example_tweet = 'I LOVE deep learning! I\'ve learned so much in this class.'
    # args.model.classify_tweet(example_tweet, vocab)

if __name__ == '__main__':
    main()
