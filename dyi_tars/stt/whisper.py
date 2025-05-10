import whisper

class Whisper:
    def __init__(self, model_path="small", device="cpu"):
        self.model = whisper.load_model(model_path)
        self.device = device
        self.model.to(self.device)

    def transcribe(self, audio):
        result = self.model.transcribe(audio)        
        return result