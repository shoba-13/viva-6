# ===== Paste between the data-download cell and the final test cell =====
# Cells 1-2 (imports + data) are given: train_file_path and test_file_path are tab-separated files.
# Make sure these imports exist in cell 1:
#   import numpy as np, pandas as pd, tensorflow as tf

# ---------------------------------------------------------------- load data
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# ham = 0, spam = 1
train_data['y'] = (train_data['label'] == 'spam').astype(int)
test_data['y'] = (test_data['label'] == 'spam').astype(int)

train_x = train_data['message'].values
train_y = train_data['y'].values
test_x = test_data['message'].values
test_y = test_data['y'].values

# ---------------------------------------------------------------- text vectorization
MAX_TOKENS = 10000
SEQ_LEN = 50

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=SEQ_LEN)
vectorizer.adapt(train_x)

# ---------------------------------------------------------------- model
# Vectorizer is inside the model, so it accepts raw strings
model = tf.keras.Sequential([
    tf.keras.Input(shape=(), dtype=tf.string),
    vectorizer,
    tf.keras.layers.Embedding(MAX_TOKENS, 32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------------------------------------------------------- train
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_x, train_y,
    epochs=30,
    validation_data=(test_x, test_y),
    callbacks=[early_stop],
    verbose=2)

print('Test accuracy:', model.evaluate(test_x, test_y, verbose=0)[1])

# ---------------------------------------------------------------- predict function
def predict_message(pred_text):
    prob = float(model.predict(tf.constant([pred_text]), verbose=0)[0][0])
    label = 'spam' if prob >= 0.5 else 'ham'
    return [prob, label]

# quick check
print(predict_message("how are you doing today?"))
print(predict_message("WINNER!! You have won a free prize, call now to claim"))
