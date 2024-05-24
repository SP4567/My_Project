from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, VitsModel, AutoTokenizer
from pydub import AudioSegment
import whisper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Load the models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Whisper model setup
whisper_model_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)

processor = AutoProcessor.from_pretrained(whisper_model_id)
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# MMS-TTS model setup
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the audio file using Whisper
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        print(f"Detected language: {detected_language}")
        options = whisper.DecodingOptions()
        result = whisper.decode(whisper_model, mel, options)
        transcribed_text = result.text

        # Convert text to speech using MMS-TTS
        inputs = tts_tokenizer(transcribed_text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform

        output_audio_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp3')
        audio_segment = AudioSegment(
            output.numpy().astype('int16').tobytes(),
            frame_rate=tts_model.config.sampling_rate,
            sample_width=2,
            channels=1
        )
        audio_segment.export(output_audio_path, format='mp3')

        return render_template('index.html', transcribed_text=transcribed_text, audio_file='output.mp3')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/outputs/<filename>')
def output_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
