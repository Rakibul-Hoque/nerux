import numpy as np
from ...tensor import Tensor
from .base import Base


class BatchNorm2D(Base):
    """
    Batch Normalization for 2D inputs (N, C, H, W).
    Normalizes across the batch dimension for each channel.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def build(self, in_shape):
        # Learnable parameters
        self.gamma = self.add_parameter(
            "gamma", Tensor(np.ones((self.num_features,)), requires_grad=True)
        )
        self.beta = self.add_parameter(
            "beta", Tensor(np.zeros((self.num_features,)), requires_grad=True)
        )

        # Running statistics (not trainable)
        self.running_mean = np.zeros((self.num_features,))
        self.running_var = np.ones((self.num_features,))

    def forward(self, x):
        # x shape: (N, C, H, W)
        if self.training:
            # Calculate batch statistics
            # Mean over (N, H, W) dimensions, keep C
            batch_mean = x.data.mean(axis=(0, 2, 3))
            batch_var = x.data.var(axis=(0, 2, 3))

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics during inference
            mean = self.running_mean
            var = self.running_var

        # Normalize: reshape for broadcasting
        mean_t = Tensor(mean.reshape(1, -1, 1, 1))
        var_t = Tensor(var.reshape(1, -1, 1, 1))

        x_normalized = (x - mean_t) / Tensor(np.sqrt(var_t.data + self.eps))

        # Scale and shift
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)

        return x_normalized * gamma + beta


"""
Example CNN architecture using all these layers:

from nerux import layers, Tensor

class CNN(layers.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = layers.Conv2D(32, kernel_size=3, padding=1)
        self.bn1 = layers.BatchNorm2d(32)
        self.pool1 = layers.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = layers.Dropout(0.25)
        
        self.conv2 = layers.Conv2D(64, kernel_size=3, padding=1)
        self.bn2 = layers.BatchNorm2d(64)
        self.pool2 = layers.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = layers.Dropout(0.25)
        
        # Fully connected layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation=layers.ReLU())
        self.dropout3 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(num_classes, activation=layers.Softmax())
    
    def forward(self, x):
        # Input: (N, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

# Usage
model = CNN(num_classes=10)
model.input(3, 32, 32)  # RGB image 32x32

# Training
model.set_training(True)
x = Tensor.randn((16, 3, 32, 32), requires_grad=True)  # Batch of 16 images
y_pred = model(x)

# Inference
model.set_training(False)
x_test = Tensor.randn((1, 3, 32, 32))
y_pred = model(x_test)
"""
