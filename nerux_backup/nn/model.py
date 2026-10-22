from .layers.layer import Layer
from ..data.dataloader import DataLoader
from .evaluation import evaluation
import pickle, os, json
import numpy as np


class Model(Layer):
    def __init__(self):
        super().__init__()

    def meta(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function

    def predict(self, input):
        input_with_batch = input.unsqueeze(axis=0)
        return self(input_with_batch)[0]

    def train(
        self,
        data,
        epochs=10,
        batch_size=16,
        shuffle=True,
        validation_data=None,
        fall_back=None
    ):
        if not hasattr(self, "optimizer"):
            raise ValueError("model must be set with an optimizer ")
        if not hasattr(self, "loss_function"):
            raise ValueError("model must be set with an loss_function ")

        training_loader = DataLoader(data, batch_size, shuffle)
        validation_loader = (
            DataLoader(validation_data, batch_size, shuffle)
            if validation_data
            else None
        )

        interrupted = False
        self.set_training(True)

        try:
            for epoch in range(epochs):
                training_loss = 0
                for x, y in training_loader:
                    pred = self(x)
                    loss = self.loss_function(pred, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    training_loss += loss.data
                msg = f"Epoch {epoch}/{epochs}: loss = {training_loss / len(training_loader)}"
                if validation_data:
                    validation_loss = 0
                    for x, y in validation_loader:
                        pred = self(x)
                        loss = self.loss_function(pred, y)
                        validation_loss += loss.data
                    msg += f" | val loss = {validation_loss / len(validation_loader)}"

                print(msg)

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            interrupted = True
        except Exception as e:
            print("⚠️ Training interrupted with error")
            print(f"Error: {e}")
            interrupted = True

        finally:
            self.set_training(False)
            if not interrupted:
                print("✅ Training completed successfully!")
            if fall_back:
                self.save(fall_back)

    def evaluate(self, dataset, file=None, batch_size=64, threshold=0.5):
        result = evaluation(self, dataset, batch_size=batch_size, threshold=threshold)
        if file is not None:
            with open(file, "w") as t:
                json.dump(result, t, indent=2)
        return result

    def save(self, file):
        data = self.parameters_dict()

        with open(file, "wb") as f:
            pickle.dump(data, f)

        size_mb = os.path.getsize(file) / (1024 * 1024)
        msg = f"✅ Model saved to {file} ({size_mb:.2f} MB)"
        print(msg)

    def load(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError(f"❌ The file '{file}' doesn’t exist.")
        data = {}
        with open(file, "rb") as f:
            data = pickle.load(f)

        self.set_parameters(data)
        msg = f"✅ Model is loaded from {file}"
        print(msg)
        return self
