import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Create a speech recognition object
recognizer = sr.Recognizer()


def transcribe_audio(path):
    """
    Recognize speech in the audio file.
    Reusable function to avoid repetition.
    """
    with sr.AudioFile(path) as source:
        audio_listened = recognizer.record(source)
        text = recognizer.recognize_google(audio_listened, language="lt-LT")
    return text


def get_large_audio_transcription_on_silence(path):
    """
    Split the large audio file into chunks based on silence
    and apply speech recognition on each chunk.
    """
    sound = AudioSegment.from_file(path)

    # Split audio on silence where it's 500 milliseconds or more
    chunks = split_on_silence(
        sound,
        min_silence_len=500,  # experiment with this value for your target audio file
        silence_thresh=sound.dBFS - 14,  # adjust per requirement
        keep_silence=500  # keep the silence for 500 milliseconds
    )

    folder_name = "audio_chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    whole_text = ""

    # Process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        # Recognize the chunk
        try:
            text = transcribe_audio(chunk_filename)
        except sr.UnknownValueError as e:
            print("Error:", str(e))
        else:
            text = f"{text.capitalize()}. "
            print(chunk_filename, ":", text)
            whole_text += text

    return whole_text


if __name__ == "__main__":
    audio_path = "Recording.wav"
    full_text = get_large_audio_transcription_on_silence(audio_path)

    with open("transcribed_text.txt", "w", encoding='utf-8') as file:
        file.write(full_text)

    print("\nFull text:", full_text)
