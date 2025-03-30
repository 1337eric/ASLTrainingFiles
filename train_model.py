import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_gesture_data(gestures_dir):
    X = []
    y = []
    
    for gesture in os.listdir(gestures_dir):
        gesture_path = os.path.join(gestures_dir, gesture)
        
        for recording in os.listdir(gesture_path):
            recording_path = os.path.join(gesture_path, recording)
            sequence = []
            
            for frame in range(30):
                frame_path = os.path.join(recording_path, f"{frame}.npy")
                if os.path.exists(frame_path):
                    frame_data = np.load(frame_path)
                    # Flatten the data regardless of original shape
                    frame_data = frame_data.flatten()
                    sequence.append(frame_data)
            
            if len(sequence) == 30:  # Only use complete sequences
                # Check if all frames in the sequence have the same shape
                if all(len(frame) == len(sequence[0]) for frame in sequence):
                    X.append(sequence)
                    y.append(gesture)
                else:
                    print(f"Skipping inconsistent sequence in {recording_path}")
    
    if not X:
        raise ValueError("No valid sequences found. Check your data collection process.")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded data shape: {X.shape}")
    print(f"Each frame has {X.shape[-1]} features")
    return X, y

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_gesture_model(gestures_dir, model_save_path):
    # Load and preprocess data
    X, y = load_gesture_data(gestures_dir)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder for inference
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    
    # Convert to float32 for better performance
    X = X.astype('float32')
    
    # Normalize the data
    X = X / np.max(np.abs(X)) if np.max(np.abs(X)) > 0 else X
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create and compile model with dynamic input shape based on actual data
    model = create_model(
        input_shape=(30, X.shape[2]),  # 30 timesteps, features from actual data
        num_classes=len(label_encoder.classes_)
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    
    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=400,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Save the model
    model.save(model_save_path)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {accuracy*100:.2f}%")
    
    return model, label_encoder

if __name__ == "__main__":
    gestures_dir = os.path.join("gestures")
    model_save_path = "gesture_model.h5"
    
    model, label_encoder = train_gesture_model(gestures_dir, model_save_path)