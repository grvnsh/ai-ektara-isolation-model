import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_preprocess(audio_file):
  y, sr = librosa.load(audio_file)
  spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
  spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
  spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
  return spectrogram

def create_ektara_detector_model(input_shape):
  model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
  ])
  
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  
  return model

def train_model(X_train, y_train, input_shape):
  model = create_ektara_detector_model(input_shape)
  model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
  return model

def evaluate_model(model, X_val, y_val):
  y_pred = model.predict(X_val)
  y_pred_binary = (y_pred > 0.5).astype(int)

  print("Accuracy:", accuracy_score(y_val, y_pred_binary))
  print("Precision:", precision_score(y_val, y_pred_binary))
  print("Recall:", recall_score(y_val, y_pred_binary))
  print("F1-score:", f1_score(y_val, y_pred_binary))

# Example case
# Assuming there is a list of audio file paths in audio_files and corresponding labels in labels
X, y = [], []
for audio_file, label in zip(audio_files, labels):
  spectrogram = load_and_preprocess(audio_file)
  X.append(spectrogram)
  y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train, X_train[0].shape)
evaluate_model(model, X_val, y_val)
