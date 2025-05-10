import whisper
import torch
import numpy as np


class Whisper:
    def __init__(self, model_path, device):
        self.model = whisper.load_model(model_path)
        self.device = device
        self.model.to(self.device)

    def transcribe(self, audio):
        audio = torch.from_numpy(audio).to(self.device)
        result = self.model.transcribe(audio)        
        return result