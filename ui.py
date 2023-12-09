import streamlit as st
import datetime
from tqdm import tqdm
from src.utils.constants import MODEL_TYPES, LANGUAGE_CODES
import src.utils.task as task

class CustomTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbar = st.progress(0, text=self.desc)

    def update_streamlit_progress(self):
        # Calculate progress percentage
        progress_percent = (self.n / self.total) * 100
        # Update Streamlit progress bar
        eta = (self.total - self.n) / self.format_dict["rate"]
        eta_str = str(datetime.timedelta(seconds=round(eta))) 
        self.pbar.progress(progress_percent / 100, text=f"Estimated remaining time: {eta_str}")

    def update(self, n=1):
        super().update(n)
        self.update_streamlit_progress()

language_names = [v for _, v in LANGUAGE_CODES]
languages = {name: code for code, name in LANGUAGE_CODES}

st.title('Whisper AutoTranscribe2')
uploaded_file = st.file_uploader("Choose an audio file", type="wav")
model_size = st.select_slider("Select the size of the model. Larger size means more accurate, but also more compute intensive", options=MODEL_TYPES, value=MODEL_TYPES[-2])
lang_name = st.selectbox("Specify audio language. Auto for auto-discovery", language_names)
lang = languages[lang_name]
audio_path = "audio.wav"
out_path = "transcript.txt"
subtitle_path = None

start_exec = st.button("Transcribe!", disabled=uploaded_file is None)

if uploaded_file is not None:
    with open (audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

if start_exec:
    with st.spinner("Transcribing, please wait..."):
        subtitle_path = task.transcribe(
            audio_path,
            subtitle=out_path,
            vocal_extracter=False,
            output_text=True,
            language=lang,
            model_type=model_size,
            transcribe_model="stable_whisper",
            task="transcribe",
            delete_tempfile=True,
            pbar_cls=CustomTqdm
        )

        with open(subtitle_path, "r") as f:
            transcript = f.read()

    st.success("Done!")
    with st.expander("Read transcription"):
        st.write(transcript)
    st.download_button('Download transcript', transcript, file_name="transcript.txt")
    uploaded_file = None