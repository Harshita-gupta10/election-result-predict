# src/models/lstm_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ElectionLSTM:
    def __init__(self, input_shape, output_dir='models/saved'):
        self.input_shape = input_shape  # (sequence_length, n_features)
        self.output_dir = output_dir
        self.model = None
        self.history = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def build_model(self, lstm_units=64, dropout_rate=0.2):
        """
        Build LSTM model architecture
        """
        model = Sequential([
            # Bidirectional LSTM layer
            Bidirectional(LSTM(
                units=lstm_units, 
                activation='tanh',
                return_sequences=True,
                input_shape=self.input_shape
            )),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            Bidirectional(LSTM(
                units=lstm_units, 
                activation='tanh', 
                return_sequences=False
            )),
            Dropout(dropout_rate),
            
            # Output layer
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the LSTM model
        """
        if self.model is None:
            self.build_model()
        
        # Define callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Checkpoint to save best model
        checkpoint_path = os.path.join(self.output_dir, f'lstm_model_{timestamp}.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # TensorBoard for visualization
        log_dir = os.path.join('logs', f'lstm_{timestamp}')
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, tensorboard],
            verbose=1
        )
        
        self.history = history.history
        return history
    
    def save_model(self, filename=None):
        """
        Save the trained model and training history
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'lstm_election_model_{timestamp}'
        
        # Save model
        model_path = os.path.join(self.output_dir, f'{filename}.h5')
        self.model.save(model_path)
        
        # Save history
        if self.history is not None:
            history_path = os.path.join(self.output_dir, f'{filename}_history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(self.history, f)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create instance
        instance = cls(input_shape=None)
        
        # Load model
        instance.model = tf.keras.models.load_model(model_path)
        
        # Get input shape from loaded model
        instance.input_shape = instance.model.input_shape[1:]
        
        return instance
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and generate metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Build and train a model first.")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Display results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        # Create a dictionary of evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'y_pred_prob': y_pred_prob.flatten().tolist(),
            'y_pred': y_pred.tolist(),
            'y_test': y_test.tolist()
        }
        
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(self.output_dir, f'evaluation_metrics_{timestamp}.json')
        
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Evaluation metrics saved to {metrics_path}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation metrics
        """
        if self.history is None:
            raise ValueError("No training history available. Train a model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history['loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def predict_election(self, X_new):
        """
        Make election predictions on new data
        
        Args:
            X_new: Preprocessed sequences in the format expected by the LSTM model
            
        Returns:
            Probabilities of election outcomes
        """
        if self.model is None:
            raise ValueError("No model for prediction. Build and train a model first.")
        
        # Make predictions
        probabilities = self.model.predict(X_new)
        
        return probabilities

# Example usage
if __name__ == "__main__":
    # Find prepared data files
    data_dir = 'data/processed'
    X_train_files = glob.glob(os.path.join(data_dir, 'X_train_*.npy'))
    
    if X_train_files:
        latest_file = max(X_train_files, key=os.path.getctime)
        timestamp = latest_file.split('_')[-1].split('.')[0]
        
        # Load the data
        X_train = np.load(os.path.join(data_dir, f'X_train_{timestamp}.npy'))
        X_test = np.load(os.path.join(data_dir, f'X_test_{timestamp}.npy'))
        y_train = np.load(os.path.join(data_dir, f'y_train_{timestamp}.npy'))
        y_test = np.load(os.path.join(data_dir, f'y_test_{timestamp}.npy'))
        
        # Create and train model
        input_shape = X_train.shape[1:]  # (sequence_length, n_features)
        model = ElectionLSTM(input_shape)
        model.build_model(lstm_units=128, dropout_rate=0.3)
        
        # Print model summary
        model.model.summary()
        
        # Train model
        history = model.train(
            X_train, y_train, 
            X_test, y_test,  # Using test data as validation for simplicity
            epochs=50, 
            batch_size=32
        )
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Plot training history
        model.plot_training_history(
            save_path=os.path.join('results', f'training_history_{timestamp}.png')
        )
        
        # Save model
        model.save_model(f'lstm_election_model_{timestamp}')
    else:
        print("No prepared data files found")