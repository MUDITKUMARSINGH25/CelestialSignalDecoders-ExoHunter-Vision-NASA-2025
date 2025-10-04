"""
ExoHunter-Vision: Multi-Modal AI for Exoplanet Detection
NASA Space Apps Challenge 2025
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ExoHunterVision:
    """
    Multi-modal AI system for exoplanet detection combining Vision Transformer 
    and temporal analysis architectures.
    """
    
    def __init__(self, image_shape=(128, 128, 3), sequence_length=1000):
        self.image_shape = image_shape
        self.sequence_length = sequence_length
        self.model = self._build_multi_modal_model()
        
    def _build_vision_branch(self):
        """Vision Transformer branch for phase-folded light curve images"""
        inputs = tf.keras.Input(shape=self.image_shape)
        
        # CNN feature extraction
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        
        return Model(inputs, x, name="vision_branch")
    
    def _build_temporal_branch(self):
        """LSTM branch for time series analysis"""
        inputs = tf.keras.Input(shape=(self.sequence_length, 1))
        
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(32)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        return Model(inputs, x, name="temporal_branch")
    
    def _build_multi_modal_model(self):
        """Combine vision and temporal branches"""
        # Inputs
        image_input = tf.keras.Input(shape=self.image_shape, name="image_input")
        temporal_input = tf.keras.Input(shape=(self.sequence_length, 1), name="temporal_input")
        
        # Process through branches
        vision_features = self._build_vision_branch()(image_input)
        temporal_features = self._build_temporal_branch()(temporal_input)
        
        # Feature fusion
        combined = layers.concatenate([vision_features, temporal_features])
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Multi-task outputs
        detection_output = layers.Dense(1, activation='sigmoid', name='detection')(x)
        period_output = layers.Dense(1, activation='linear', name='period')(x)
        depth_output = layers.Dense(1, activation='linear', name='depth')(x)
        
        model = Model(
            inputs=[image_input, temporal_input],
            outputs=[detection_output, period_output, depth_output]
        )
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with multi-task learning"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'detection': 'binary_crossentropy',
                'period': 'mse',
                'depth': 'mse'
            },
            loss_weights={'detection': 1.0, 'period': 0.3, 'depth': 0.3},
            metrics={'detection': ['accuracy', 'precision', 'recall']}
        )
    
    def summary(self):
        """Print model architecture"""
        return self.model.summary()

# Example usage
if __name__ == "__main__":
    model = ExoHunterVision()
    model.compile_model()
    model.summary()
