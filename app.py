import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForSeq2SeqLM
import gradio as gr
from datasets import load_dataset

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor for transcription
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
processor.feature_extractor.sampling_rate = 16000  # Ensure the sampling rate is set correctly

# Set up the transcription pipeline
transcription_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Load the translation model and set up the translation pipeline
translation_model_id = "Helsinki-NLP/opus-mt-hi-en"
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_id)
translation_pipe = pipeline("translation_hi_to_en", model=translation_model, tokenizer=translation_model_id,
                            device=device)


# Function to transcribe and translate
def transcribe_and_translate(audio):
    if audio is None:
        return "Please upload an audio file.", ""

    try:
        # Transcription
        result = transcription_pipe(audio)
        text = result["text"]

        # Translation
        translation_result = translation_pipe(text)
        translation = translation_result[0]["translation_text"]

        return text, translation
    except Exception as e:
        return f"An error occurred: {e}", ""


# Gradio interface setup
inputs = gr.Audio(type="filepath", label="Upload Audio File")
outputs = [gr.Textbox(label="Transcription"), gr.Textbox(label="English Translation")]

examples = [
    ["/content/urdu.mp3"]  # Replace with the actual path to your urdu.mp3 file
]

app = gr.Interface(
    fn=transcribe_and_translate,
    inputs=inputs,
    outputs=outputs,
    title="Audio Transcription and Translation",
    examples=examples,
)
app.launch(share=True)
