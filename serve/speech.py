"""serve/speech.py - text to speech for a Drama 3 engine (guild-generate --drama3): voices, WAV, request shapes.

The engine speaks with two pipe commands (src/program/generate.cpp):

    VOICE <sample rate> <base64 float32 samples>  ->  CODES <frames> <comma-separated codes>
    SAY [max_new=] [temperature=] [top_p=] [top_k=] [seed=] [chunk=] text=<base64> [ref=<base64 text>:<codes>]...
        ->  A <base64 16-bit PCM> per chunk, then  SDONE <frames> <prompt rows> <prompt ms> <decode ms> <reason> <rate>

A voice is a reference clip and its transcript.  The server keeps named voices in a folder (`voices_dir` in the run
config, or --voices): <name>.wav with <name>.txt beside it; the clip's codes are computed once (VOICE) and kept in
<name>.codes.  A request may also bring its own clip (Fish Audio's /v1/tts `references`).  Only the standard
library is used here: WAV through `wave`, samples through `array`.
"""
from __future__ import annotations

import array
import io
import re
import secrets
import struct
import sys
import threading
import wave
from pathlib import Path

MAX_REFERENCE_S = 60.0   # the reference implementation's cap on a reference clip
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def wav_samples(data: bytes) -> tuple[bytes, int]:
    """A WAV file's samples as little-endian float32 mono (channels averaged), and its sample rate.  PCM 8/16/24/32"""
    try:
        w = wave.open(io.BytesIO(data))
    except (wave.Error, EOFError) as e:
        raise ValueError(f"reference audio: not a PCM WAV file ({e})") from None
    with w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if not 4000 <= rate <= 384000:
            raise ValueError(f"reference audio: sample rate {rate} Hz")
        # what is there, not the header's count: a streamed WAV says 0xFFFFFFFF bytes
        raw = w.readframes(int(MAX_REFERENCE_S * rate) + 1)
        if len(raw) // (width * ch) > MAX_REFERENCE_S * rate:
            raise ValueError(f"reference audio: longer than {MAX_REFERENCE_S:.0f} s")
    if width == 1:
        vals = [(b - 128) / 128.0 for b in raw]
    elif width == 2:
        a = array.array("h", raw)
        if sys.byteorder == "big":
            a.byteswap()
        vals = [v / 32768.0 for v in a]
    elif width == 3:
        vals = [int.from_bytes(raw[i:i + 3], "little", signed=True) / 8388608.0 for i in range(0, len(raw), 3)]
    elif width == 4:
        a = array.array("i", raw)
        if sys.byteorder == "big":
            a.byteswap()
        vals = [v / 2147483648.0 for v in a]
    else:
        raise ValueError(f"reference audio: {8 * width}-bit samples are not supported")
    if ch > 1:
        vals = [sum(vals[i:i + ch]) / ch for i in range(0, len(vals), ch)]
    out = array.array("f", vals)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes(), rate


def wav_header(rate: int, n_samples: int | None) -> bytes:
    """A 16-bit mono WAV header; n_samples None = a stream of unknown length (sizes 0xFFFFFFFF)"""
    data = 0xFFFFFFFF if n_samples is None else 2 * n_samples
    riff = 0xFFFFFFFF if n_samples is None else 36 + data
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, 2 * rate, 2, 16) + b"data" + struct.pack("<I", data))


class Voices:
    """The named voices of a folder: <name>.wav + <name>.txt, their codes cached in <name>.codes"""

    def __init__(self, folder: str | None):
        self.folder = Path(folder) if folder else None
        self.lock = threading.Lock()

    def names(self) -> list[str]:
        if self.folder is None or not self.folder.is_dir():
            return []
        return sorted(p.stem for p in self.folder.glob("*.wav") if NAME.match(p.stem) and
                      p.with_suffix(".txt").is_file())

    def reference(self, name: str, encode) -> tuple[str, str]:
        """(transcript, codes) of voice `name`; `encode(float32 bytes, rate) -> codes` runs once per clip"""
        if not NAME.match(name or "") or name not in self.names():
            raise ValueError(f"unknown voice {name!r}; the voices are {', '.join(self.names()) or '(none)'}")
        wav, txt = self.folder / f"{name}.wav", self.folder / f"{name}.txt"
        cached = self.folder / f"{name}.codes"
        text = txt.read_text(encoding="utf-8").strip()
        with self.lock:
            if cached.is_file() and cached.stat().st_mtime >= wav.stat().st_mtime:
                return text, cached.read_text(encoding="ascii").strip()
            samples, rate = wav_samples(wav.read_bytes())
            codes = encode(samples, rate)
            try:
                cached.write_text(codes + "\n", encoding="ascii")
            except OSError:
                pass                                   # a read-only folder: encode again next time
            return text, codes


SPEAKER = re.compile(r"^\s*<\|speaker:[0-9]+\|>")


def with_direction(text: str, direction: str) -> str:
    """`text` with a bracketed direction before its line - after a leading <|speaker:N|> tag, which must stay first
    (the engine adds <|speaker:0|> to a text that does not start with one)"""
    direction = direction.strip().replace("[", "(").replace("]", ")")
    if not direction:
        return text
    m = SPEAKER.match(text)
    if m:
        return f"{text[:m.end()]} [{direction}] {text[m.end():].lstrip()}"
    return f"[{direction}] {text}"


def say_line(text: str, refs: list[tuple[str, str]], params: dict) -> str:
    """The SAY command for `text`, the reference voices (transcript, codes) and the sampling `params`"""
    import base64

    def b64(s: str) -> str:
        return base64.b64encode(s.encode("utf-8")).decode("ascii")

    keys = []
    for k, cast in (("max_new", int), ("temperature", float), ("top_p", float), ("top_k", int), ("seed", int),
                    ("chunk", int)):
        if params.get(k) is not None:
            keys.append(f"{k}={cast(params[k])}")
    keys.append(f"text={b64(text)}")
    keys += [f"ref={b64(t)}:{c}" for t, c in refs]
    return "SAY " + " ".join(keys)


def speech_params(req: dict) -> dict:
    """The sampling of a speech request (both APIs' names); unset keys keep the model card's (1.0 / 0.9 / 30), and
    an unset seed is drawn at random"""
    out = {}
    for key, names in (("temperature", ("temperature",)), ("top_p", ("top_p",)), ("top_k", ("top_k",)),
                       ("seed", ("seed",)), ("max_new", ("max_new_tokens", "max_frames"))):
        for n in names:
            v = req.get(n)
            if v is not None:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"),
                                                                                             float("-inf")):
                    raise ValueError(f"{n}: a finite number")
                out[key] = v
    if "top_k" in out and int(out["top_k"]) < 1:
        raise ValueError("top_k: at least 1")
    if "top_p" in out and not 0 < out["top_p"] <= 1:
        raise ValueError("top_p: in (0, 1]")
    if "temperature" in out and out["temperature"] < 0:
        raise ValueError("temperature: 0 or more")
    if "max_new" in out and not 1 <= int(out["max_new"]) <= 8192:
        raise ValueError("max_new_tokens: 1..8192 frames (21.5 per second of audio)")
    if "seed" not in out:          # a fresh take each time; the seed goes back in X-Guild-Seed to replay it
        out["seed"] = secrets.randbits(63)
    elif not 0 <= int(out["seed"]) < 2 ** 64:
        raise ValueError("seed: 0 .. 2^64 - 1")
    return out
