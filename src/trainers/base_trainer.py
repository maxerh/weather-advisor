import torch
import torch.nn as nn
import mlflow
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any



class BaseTrainer(ABC):
    """
    Abstract Base Trainer class providing:
    - training loop
    - validation loop
    - early stopping
    - MLflow logging support
    - model saving

    Subclasses only need to implement:
      - build_model()
      - compute_loss(outputs, targets)
      - predict_step(batch)
    """

    def __init__(
            self,
            model_name: str,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            learning_rate: float = 1e-3,
            epochs: int = 20,
            patience: int = 5,
            save_dir: str = "trained_models/",
            mlflow_experiment: str = "weather_forecasting",
    ):
        self.model_name = model_name
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.save_dir = Path(save_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self.model = self.build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

        if not hasattr(self, 'input_length'):
            self.input_length = None
        if not hasattr(self, 'hidden_dim'):
            self.hidden_dim = None
        if not hasattr(self, 'num_layers'):
            self.num_layers = None
        if not hasattr(self, 'embed_dim'):
            self.embed_dim = None
        if not hasattr(self, 'num_heads'):
            self.num_heads = None
        if not hasattr(self, 'ff_dim'):
            self.ff_dim = None

        # MLflow init
        mlflow.set_experiment(mlflow_experiment)

    # --------------------------------------------------------
    # ABSTRACT METHODS — MUST BE IMPLEMENTED IN CHILD CLASSES
    # --------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return a PyTorch model."""
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets) -> torch.Tensor:
        """Compute the loss for one batch."""
        pass

    @abstractmethod
    def predict_step(self, batch):
        """Forward pass for one batch."""
        pass

    # --------------------------------------------------------
    # TRAINING + VALIDATION LOOP
    # --------------------------------------------------------

    def train(self):
        """Full training loop with MLflow logging and early stopping."""

        with mlflow.start_run():
            mlflow.log_param("model", self.model_name)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("patience", self.patience)

            for epoch in range(1, self.epochs + 1):
                train_loss = self.train_one_epoch()
                val_loss = self.validate_one_epoch()

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                print(f"Epoch [{epoch}/{self.epochs}]  "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # Check early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    self.save_model()  # save best model
                else:
                    self.early_stopping_counter += 1

                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            # Log best model file
            mlflow.log_artifact(str(self.save_dir / f"{self.model_name}.pt"))

    # --------------------------------------------------------
    # ONE EPOCH OF TRAINING
    # --------------------------------------------------------

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            inputs, targets = [x.to(self.device) for x in batch]

            outputs = self.predict_step(inputs)
            loss = self.compute_loss(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # --------------------------------------------------------
    # ONE EPOCH OF VALIDATION
    # --------------------------------------------------------

    def validate_one_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = [x.to(self.device) for x in batch]

                outputs = self.predict_step(inputs)
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    def save_model(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.save_dir / f"{self.model_name}.pt"
        #torch.save(self.model.state_dict(), model_path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'input_length': self.input_length,
            'output_dim': self.output_dim,
            'pred_len': self.pred_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'embed_dim' : self.embed_dim,
            'num_heads' : self.num_heads,
            'ff_dim' : self.ff_dim,
        }, model_path)
        print(f"Saved best model → {model_path}")