from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.callbacks import ReduceLROnPlateau
import gensim
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")
# Cleaning data
unknown_publishers = []
for index, row in enumerate(real.text.values):
    try:
        record = row.split(" -", maxsplit=1)
        assert(len(record[0]) < 260)
    except:
        unknown_publishers.append(index)
publisher = []
tmp_text = []
for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append("Unknown")
        continue
    record = row.split(" -", maxsplit=1)
    publisher.append(record[0])
    tmp_text.append(record[1])
real["publisher"] = publisher
real["text"] = tmp_text
real = real.drop(8970, axis=0)
real["class"] = 1
fake["class"] = 0
real["text"] = real["title"] + " " + real["text"]
fake["text"] = fake["title"] + " " + fake["text"]
real = real.drop(["subject", "date", "title",  "publisher"], axis=1)
fake = fake.drop(["subject", "date", "title"], axis=1)
data = real.append(fake, ignore_index=True)
# Word Cloud
text = ''
for news in data.text.values:
    text += f" {news}"
wordcloud = WordCloud(
    width=3000,
    height=2000,
    background_color='black',
    stopwords=set(nltk.corpus.stopwords.words("english"))).generate(text)
fig = plt.figure(
    figsize=(40, 30),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('wc.png', dpi=300)


# Set X and y
y = data["class"].values
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip()
                          for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

# Training Word2vec
EMBEDDING_DIM = 100  # can be 200, 300
w2v_model = gensim.models.Word2Vec(
    sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)
# Tokenizing Text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Padding and Truncate
maxlen = 700
X = pad_sequences(X, maxlen=maxlen)
vocab_size = len(tokenizer.word_index) + 1  # for unknown words


def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


embedding_vectors = get_weight_matrix(w2v_model, word_index)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 10
embed_size = 100
embedding_vectors = get_weight_matrix(w2v_model, word_index)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 10
embed_size = 100

# Defining Neural Network
model = Sequential()
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[
          embedding_vectors], input_length=maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Training
history = model.fit(X_train, y_train, validation_split=0.3,
                    epochs=epochs, callbacks=[learning_rate_reduction])
y_pred = (model.predict(X_test) >= 0.5).astype("int")
# Pictures
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm, index=['Fake', 'Original'], columns=['Fake', 'Original'])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True,
            fmt='', xticklabels=['Fake', 'Original'], yticklabels=['Fake', 'Original'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('cm.png', dpi=300)

# Picture of comparation
with plt.style.context(('seaborn-whitegrid')):
    epochs = [i for i in range(10)]
    fig, ax = plt.subplots(figsize=(8, 6))
    train_acc1 = [i for i in history1['acc']]
    train_acc2 = [i for i in history2['acc']]
    train_acc3 = [i for i in history3['acc']]
    train_acc4 = [i for i in history4['acc']]
    train_acc5 = [i for i in history5['acc']]
    train_acc6 = [i for i in history6['acc']]
    train_acc7 = [i for i in history7['acc']]
    train_acc8 = [i for i in history8['acc']]
    train_acc9 = [i for i in history9['acc']]
    train_acc10 = [i for i in history10['acc']]
    train_acc11 = [i for i in history11['acc']]
    train_acc12 = [i for i in history12['acc']]

    val_acc1 = [i for i in history1['val_acc']]
    val_acc2 = [i for i in history2['val_acc']]
    val_acc3 = [i for i in history3['val_acc']]
    val_acc4 = [i for i in history4['val_acc']]
    val_acc5 = [i for i in history5['val_acc']]
    val_acc6 = [i for i in history6['val_acc']]
    val_acc7 = [i for i in history7['val_acc']]
    val_acc8 = [i for i in history8['val_acc']]
    val_acc9 = [i for i in history9['val_acc']]
    val_acc10 = [i for i in history10['val_acc']]
    val_acc11 = [i for i in history11['val_acc']]
    val_acc12 = [i for i in history12['val_acc']]
    fig.set_size_inches(20, 10)

    ax[0].plot(epochs, train_acc1, label='100 dimensions + CBOW + Negative Sampling')
    ax[0].plot(epochs, train_acc2,
            label='100 dimensions + CBOW + Hierarchical Softmax')
    ax[0].plot(epochs, train_acc3,
            label='100 dimensions + Skip-gram + Negative Sampling')
    ax[0].plot(epochs, train_acc4,
            label='100 dimensions + Skip-gram + Hierarchical Softmax')
    ax[0].plot(epochs, train_acc5, label='200 dimensions + CBOW + Negative Sampling')
    ax[0].plot(epochs, train_acc6,
            label='200 dimensions + CBOW + Hierarchical Softmax')
    ax[0].plot(epochs, train_acc7,
            label='200 dimensions + Skip-gram + Negative Sampling')
    ax[0].plot(epochs, train_acc8,
            label='200 dimensions + Skip-gram + Hierarchical Softmax')
    ax[0].plot(epochs, train_acc9, label='300 dimensions + CBOW + Negative Sampling')
    ax[0].plot(epochs, train_acc10,
            label='300 dimensions + CBOW + Hierarchical Softmax')
    ax[0].plot(epochs, train_acc11,
            label='300 dimensions + Skip-gram + Negative Sampling')
    ax[0].plot(epochs, train_acc12,
            label='300 dimensions + Skip-gram + Hierarchical Softmax')

    ax[1].plot(epochs, val_acc1, label='Testing Accuracy1')
    ax[1].plot(epochs, val_acc2, label='Testing Accuracy2')
    ax[1].plot(epochs, val_acc3, label='Testing Accuracy3')
    ax[1].plot(epochs, val_acc4, label='Testing Accuracy4')
    ax[1].plot(epochs, val_acc5, label='Testing Accuracy5')
    ax[1].plot(epochs, val_acc6, label='Testing Accuracy6')
    ax[1].plot(epochs, val_acc7, label='Testing Accuracy7')
    ax[1].plot(epochs, val_acc8, label='Testing Accuracy8')
    ax[1].plot(epochs, val_acc9, label='Testing Accuracy9')
    ax[1].plot(epochs, val_acc10, label='Testing Accuracy10')
    ax[1].plot(epochs, val_acc11, label='Testing Accuracy11')
    ax[1].plot(epochs, val_acc12, label='Testing Accuracy12')

    ax[0].set_title('Training Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_title('Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=15)
    plt.tight_layout()
plt.savefig('trainACC_all.png', dpi=300)
