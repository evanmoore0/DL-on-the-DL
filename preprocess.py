import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print('Manipulating the dataframe...')

# Importing the dataset
columns = ["target", "ids", "date", "flag", "user", "text"]
df_all = pd.read_csv('./data/twitter_data.csv', encoding="ISO-8859-1",
                     names=columns)

# Keeping only the label and the tweets
unnecessary_columns = ['ids', 'date', 'flag', 'user']
df_all.drop(unnecessary_columns, axis=1, inplace=True)

# Changing positive sentiment label to be 1 instead of 4
df_all['target'] = df_all['target'].replace([4], 1)

# We want to work only on the sample of our data
df = df_all.sample(500000, random_state=42)  # 49.875% of tweets are negative, so we preserve the balance of labels

# Defining a set of stopwords to use later
stop_words = set(stopwords.words('english'))

# Exporting the 'text' column to a list so that it's easier to manipulate words
text = df['text'].tolist()

lemmatizer = WordNetLemmatizer()
lemmatized_text = []

print('Lemmatizing and removing stop words...')

for sent in text:
    words = sent.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    sentence = ' '.join(words)
    lemmatized_text.append(sentence)

target = df['target'].tolist()

print('Splitting the data...')

# 70/30 train/test split
X_train, X_test, y_train, y_test = train_test_split(lemmatized_text, target, test_size=0.3, random_state=42,
                                                    stratify=target)

print('Tokenizing the data...')

# Tokenizing the data

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=1000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ',
    oov_token='<UNK>',
)

print('Fitting the tokenizer on text and finding vocabulary size...')

tokenizer.fit_on_texts(X_train)
train_unique_words = sorted(set(X_train))
vocab_sz = len(train_unique_words)

print("VOCAB SIZE")
print(vocab_sz)

print('Transforming text to sequences and padding the sequences...')
# Transforming text to sequences and padding the sequences
train_sequence = tokenizer.texts_to_sequences(X_train)
padded_train_sequence = tf.keras.preprocessing.sequence.pad_sequences(train_sequence, maxlen=100, padding='post',
                                                                      truncating='post')
test_sequence = tokenizer.texts_to_sequences(X_test)
padded_test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen=100, padding='post',
                                                                     truncating='post')

padded_train_sequence = np.array(padded_train_sequence)
padded_test_sequence = np.array(padded_test_sequence)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#print(1, X_test[0:5])


#vectorizing the words in each sentence creating a vector of sentences that are a vector of words
X_test = np.array([nltk.word_tokenize(sentence) for sentence in X_test])


#add tags to each word in a sentence that reprents its form
#output is a vector of sentences with each word being in a tuple of the word and the tag
X_test = np.array([nltk.pos_tag(sentence) for sentence in X_test])


#part of speech tags associated with verbs
#full list of part-of-speech tags https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

#remove verbs from sentences based on tags
X_test = [[tag_nested for tag_nested in tag if tag_nested[1] not in verb_tags] for tag in X_test]

#unpacks the tuples to only retain the words returning an array of str
X_test = [[tup[0] for tup in sentence] for sentence in X_test]
#print(5, X_test[0:2])

#combines 
X_test = [" ".join(sentence) for sentence in X_test]
#print(6, X_test[0:5])

