import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
data = pd.read_csv('twitter_training.csv')
data.head()
x = data.iloc[:,-1].values
y = data.iloc[:,-2].values
print(x)
print(type(x))
print(x.shape)
print(type(y))
print(y)
print(y.shape)
# to see the type of sentiments.....
a = data.iloc[:, -2].value_counts()
print(a)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
for index, class_label in enumerate(label_encoder.classes_):
    print(f"Class: {class_label}, Number: {index}")
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = X_train.astype(str)
X_test = X_test.astype(str)
tokenizer = Tokenizer(num_words=10000)  # Keep top 10,000 words
tokenizer.fit_on_texts(X_train)
print(len(tokenizer.word_index))
X_train_seq = tokenizer.texts_to_sequences(X_train)  # Convert to sequences
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq,120,  padding='post')
X_test_padded = pad_sequences(X_test_seq,120, padding='post')
print(X_test_padded.shape)
print(X_train_padded.shape)
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=120),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_padded, y_train,
    validation_split=0.2,
    epochs=1,
    batch_size=32
)
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences(text)
    padd = pad_sequences(sequence,120,padding='post')
    predict = model.predict(padd)
    p_class = np.argmax(predict)
    label = ['irrelevant','negative','neutral','positive']
    print(label[p_class])


text = 'Pccoe is the best college in pune'
predict_sentiment(text)
