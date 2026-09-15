"""
Voice layer: speak the question, listen to the answer, judge the delivery.

Interview line: "Voice runs on the client side, next to the microphone and the
speakers, so the backend stays a pure text API. Both halves are swappable the
same way the LLM is - one env var each.

  TTS  gTTS by default, natural sounding, needs internet.
       pyttsx3 is the offline fallback and it is automatic: if gTTS throws,
       the question still gets read aloud.

  STT  faster-whisper by default. It runs locally, needs no key, and handles
       Indian-accented English noticeably better than the Google Web Speech
       endpoint. SpeechRecognition is still in the loop, but only for the part
       it is genuinely good at - opening the microphone and deciding when I
       have stopped talking. It writes a WAV, Whisper transcribes it.

The bit worth pointing at is analyse_speaking. Most mock-interview projects
score only what you said; this one also scores how you said it - filler-word
rate and answer length become a fluency score that travels to the backend
alongside the content score."

Every import is lazy, inside the function, so the app still starts on a machine
with no microphone and no audio libraries installed.
"""

import os
import re
import tempfile
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "gtts").lower()
STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper").lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "actually", "basically",
    "literally", "i mean", "sort of", "kind of", "so yeah", "right",
]


# ---------------------------------------------------------------- text to speech
def _speak_gtts(text: str) -> bool:
    """Google TTS: write an mp3 to a temp file and play it with pygame."""
    from gtts import gTTS
    import pygame

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name
    gTTS(text=text, lang="en", tld="co.in").save(path)

    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    pygame.mixer.quit()
    os.remove(path)
    return True


def _speak_pyttsx3(text: str) -> bool:
    """Offline TTS. No network, no temp file."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    return True


def speak(text: str) -> bool:
    """Read text aloud. Tries the configured engine, falls back, never raises."""
    order = [_speak_gtts, _speak_pyttsx3] if TTS_PROVIDER == "gtts" else [_speak_pyttsx3, _speak_gtts]
    for engine in order:
        try:
            return engine(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[voice] {engine.__name__} failed ({exc})")
    print("[voice] no tts engine available, question not read aloud")
    return False


# ---------------------------------------------------------------- speech to text
@lru_cache(maxsize=1)
def _whisper():
    """
    Load the Whisper model once per process, not once per answer.

    'base' int8 on CPU is ~140MB and transcribes a 60 second answer in a few
    seconds, which is fast enough to sit inside an interview turn.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def _transcribe_whisper(path: str) -> str:
    segments, _info = _whisper().transcribe(path, language="en", beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


def _transcribe_google(path: str) -> str:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


def transcribe_file(path: str) -> str:
    """Transcribe a WAV file. Works with no microphone attached."""
    order = (
        [_transcribe_whisper, _transcribe_google]
        if STT_PROVIDER == "whisper"
        else [_transcribe_google, _transcribe_whisper]
    )
    for backend in order:
        try:
            text = backend(path)
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            print(f"[voice] {backend.__name__} failed ({exc})")
    return ""


def listen(timeout: int = 8, phrase_time_limit: int = 90) -> str:
    """
    Record one spoken answer and return the transcript.

    SpeechRecognition only does the microphone work here: ambient-noise
    calibration and deciding when the candidate has stopped speaking. The audio
    it captures is written to a WAV and handed to the configured STT backend.
    """
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except Exception as exc:  # noqa: BLE001
        print(f"[voice] microphone capture failed ({exc})")
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio.get_wav_data())
        path = tmp.name
    try:
        return transcribe_file(path)
    finally:
        os.remove(path)


# ---------------------------------------------------------------- speaking analysis
def analyse_speaking(transcript: str) -> dict:
    """
    Score delivery from the transcript alone.

    Two signals: how often filler words appear, and whether the answer had
    enough substance to be worth listening to.
    """
    words = re.findall(r"[a-z']+", (transcript or "").lower())
    word_count = len(words)
    if word_count == 0:
        return {"word_count": 0, "filler_count": 0, "filler_rate": 0.0,
                "fluency_score": 0.0, "note": "Nothing was recorded."}

    lowered = " " + " ".join(words) + " "
    filler_count = sum(lowered.count(f" {filler} ") for filler in FILLER_WORDS)
    filler_rate = filler_count / word_count

    # start at 1.0, lose points for fillers, lose points for a very short answer
    fluency = 1.0 - min(filler_rate * 4, 0.6)
    if word_count < 25:
        fluency -= 0.2
    fluency = round(max(0.0, min(1.0, fluency)), 3)

    if filler_rate > 0.06:
        note = "High filler-word rate. Pause silently instead of saying 'um'."
    elif word_count < 25:
        note = "Answer was very short. Add a concrete example."
    else:
        note = "Clear, steady delivery."

    return {
        "word_count": word_count,
        "filler_count": filler_count,
        "filler_rate": round(filler_rate, 3),
        "fluency_score": fluency,
        "note": note,
    }
