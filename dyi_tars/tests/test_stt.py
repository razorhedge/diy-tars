from dyi_tars.stt.whisper import Whisper

def test_whisper(audio_path="../samples/Recording0001.wav"):
    whisper = Whisper(model_path="small", device="cpu")
    result = whisper.transcribe(audio_path)
    print(result)
    assert result['text'] == " Hello, how are you?"