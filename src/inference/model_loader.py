import torch

class ForecastModelWrapper:
    def __init__(self, model_class, checkpoint_path, model_kwargs):
        self.model = model_class(**model_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            return self.model(x)