import torch
import torchaudio

# Load Tacotron2 TTS pipeline
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to("cpu")
vocoder = bundle.get_vocoder().to("cpu")

# Input text
text = "Hello, this is a text to speech system built using Tacotron two."

# Convert text → tokens
tokens, lengths = processor(text)

# Generate spectrogram
with torch.no_grad():
    spectrogram, spec_lengths, _ = tacotron2.infer(tokens, lengths)

# Convert spectrogram → waveform
with torch.no_grad():
    waveforms, lengths = vocoder(spectrogram, spec_lengths)

waveform = waveforms[0]

# Save audio
torchaudio.save("speech.wav", waveform.unsqueeze(0), 22050)

print("Speech generated and saved as speech.wav")