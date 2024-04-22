import requests
import pyaudio
import soundfile as sf
import io
import time





class audio_node:


    def streamed_audio(self,input_text, model='tts-1', voice='shimmer'):
        start_time = time.time()
        # OpenAI API endpoint and parameters
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": 'Bearer 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',  # Replace with your API key
        }

        data = {
            "model": model,
            "input": input_text,
            "voice": voice,
            "response_format": "opus"
        }

        audio = pyaudio.PyAudio()

        def get_pyaudio_format(subtype):
            if subtype == 'PCM_16':
                return pyaudio.paInt16
            return pyaudio.paInt16

        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                buffer = io.BytesIO()
                for chunk in response.iter_content(chunk_size=4096):
                    buffer.write(chunk)

                buffer.seek(0)

                with sf.SoundFile(buffer, 'r') as sound_file:
                    format = get_pyaudio_format(sound_file.subtype)
                    channels = sound_file.channels
                    rate = sound_file.samplerate

                    stream = audio.open(format=format, channels=channels, rate=rate, output=True)
                    chunk_size = 1024
                    data = sound_file.read(chunk_size, dtype='int16')

                    while len(data) > 0:
                        stream.write(data.tobytes())
                        data = sound_file.read(chunk_size, dtype='int16')

                    stream.stop_stream()
                    stream.close()
            else:
                self.logger.log(f"Error: {response.status_code} - {response.text}")

            audio.terminate()



