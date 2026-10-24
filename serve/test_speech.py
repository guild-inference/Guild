"""serve/test_speech.py - text to speech (serve/speech.py, Service.speak) without an engine: WAV in and out, the SAY
line the engine reads, the sampling keys, the voices folder and its codes cache, and a text model's refusal.

    python -m unittest serve.test_speech
"""
from __future__ import annotations

import array
import base64
import io
import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from serve import speech
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]


def wav_bytes(samples: list[int], rate: int = 24000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array.array("h", samples).tobytes())
    return buf.getvalue()


class SpeechEngine:
    """Answers VOICE with fixed codes and SAY with two chunks, recording what it was sent"""

    speech = True
    max_context = 10240

    def __init__(self):
        self.encoded, self.said = [], []

    def alive(self):
        return True

    def encode_voice(self, samples: bytes, rate: int) -> str:
        self.encoded.append((len(samples) // 4, rate))
        return ",".join(["1"] * 10)

    def speak(self, say: str, cancel):
        self.said.append(say)
        yield None
        yield b"\x01\x00" * 4
        yield b"\x02\x00" * 4


def service(engine) -> Service:
    tok = ByteTokenizer()
    return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


class Wav(unittest.TestCase):
    def test_mono_16(self):
        samples, rate = speech.wav_samples(wav_bytes([0, 16384, -32768]))
        self.assertEqual(rate, 24000)
        self.assertEqual(list(struct.unpack("<3f", samples)), [0.0, 0.5, -1.0])

    def test_stereo_is_averaged(self):
        samples, _ = speech.wav_samples(wav_bytes([16384, 0, -16384, -16384], channels=2))
        self.assertEqual(list(struct.unpack("<2f", samples)), [0.25, -0.5])

    def test_not_a_wav(self):
        with self.assertRaises(ValueError):
            speech.wav_samples(b"ID3 not audio")

    def test_header(self):
        h = speech.wav_header(44100, 10)
        self.assertEqual(len(h), 44)
        with wave.open(io.BytesIO(h + b"\0" * 20)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()), (44100, 1, 2, 10))
        self.assertEqual(speech.wav_header(44100, None)[4:8], b"\xff\xff\xff\xff")   # a stream: open-ended


class SayLine(unittest.TestCase):
    def test_keys_and_base64(self):
        line = speech.say_line("Hi there", [("ref text", "1,2,3")], {"temperature": 0.8, "seed": 3, "chunk": 16})
        keys = dict(k.split("=", 1) for k in line.split()[1:])
        self.assertTrue(line.startswith("SAY "))
        self.assertEqual(base64.b64decode(keys["text"]).decode(), "Hi there")
        self.assertEqual(keys["temperature"], "0.8")
        self.assertEqual(keys["seed"], "3")
        transcript, codes = keys["ref"].split(":")
        self.assertEqual((base64.b64decode(transcript).decode(), codes), ("ref text", "1,2,3"))

    def test_params(self):
        self.assertEqual(speech.speech_params({"temperature": 0.7, "max_new_tokens": 300, "voice": "x"}),
                         {"temperature": 0.7, "max_new": 300})
        for bad in ({"top_k": 0}, {"temperature": "hot"}, {"max_new_tokens": 0}, {"seed": True}):
            with self.assertRaises(ValueError, msg=bad):
                speech.speech_params(bad)


class Voices(unittest.TestCase):
    def test_folder_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "anna.wav").write_bytes(wav_bytes([0] * 100))
            (Path(d) / "anna.txt").write_text("Hello, I am Anna.\n", encoding="utf-8")
            (Path(d) / "lonely.wav").write_bytes(wav_bytes([0] * 10))       # no transcript: not a voice
            v = speech.Voices(d)
            self.assertEqual(v.names(), ["anna"])
            calls = []

            def encode(samples, rate):
                calls.append(rate)
                return "5,5"

            self.assertEqual(v.reference("anna", encode), ("Hello, I am Anna.", "5,5"))
            self.assertEqual(v.reference("anna", encode), ("Hello, I am Anna.", "5,5"))
            self.assertEqual(calls, [24000])                                 # the codes were kept in anna.codes
            for bad in ("lonely", "../anna", ""):
                with self.assertRaises(ValueError):
                    v.reference(bad, encode)


class Speak(unittest.TestCase):
    def test_references_reach_the_engine(self):
        eng = SpeechEngine()
        svc = service(eng)
        out = list(svc.speak("Hello", None, [(wav_bytes([0] * 48), "ref words")], {"seed": 1}, threading.Event()))
        self.assertEqual(b"".join(c for c in out if c), b"\x01\x00" * 4 + b"\x02\x00" * 4)
        self.assertEqual(eng.encoded, [(48, 24000)])
        keys = dict(k.split("=", 1) for k in eng.said[0].split()[1:])
        self.assertEqual(keys["chunk"], "16")
        self.assertEqual(base64.b64decode(keys["ref"].split(":")[0]).decode(), "ref words")

    def test_text_model_refuses(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        with self.assertRaisesRegex(ValueError, "does not speak"):
            list(svc.speak("Hello", None, [], {}, threading.Event()))

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            list(service(SpeechEngine()).speak("  ", None, [], {}, threading.Event()))


if __name__ == "__main__":
    unittest.main()
