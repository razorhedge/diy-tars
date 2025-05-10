import piper
import torch
import numpy as np


class Piper:
    def __init__(self, model_path, device):
        self.model = piper.load_model(model_path)
        self.device = device
        self.model.to(self.device)

    def transcribe(self, audio):
        audio = torch.from_numpy(audio).to(self.device)
        result = self.model.transcribe(audio)        
        return result