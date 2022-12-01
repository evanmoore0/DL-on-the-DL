from preprocess import y_test, y_train, padded_test_sequence, padded_train_sequence

import tensorflow as tf
import matplotlib.pyplot as plt



# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(100,)),
    tf.keras.layers.Embedding(input_dim=1000, output_dim=100, input_length=100),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(padded_train_sequence, y_train, epochs=5, batch_size = 100, validation_data=(padded_test_sequence, y_test))

# Plot the training and validation accuracy and loss at each epoch

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on the test set
model.evaluate(padded_test_sequence, y_test)

# Predict on the first 5 test examples
model.predict(padded_test_sequence[:5])

# Print the actual labels for those 5 examples
y_test[:5]

