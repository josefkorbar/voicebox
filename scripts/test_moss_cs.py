# Rychly test cestiny v MOSS-TTS (mimo Voicebox). Spusteni:
#   uv run --python 3.12 --with mlx-audio==0.6.3 python test_moss_cs.py
import glob, os
import numpy as np
from mlx_audio.tts import load
from mlx_audio.audio_io import write

TEXT = ("Jeskynní potápěči objevili v Amatérské jeskyni v Moravském krasu místo, "
        "kde se spojuje Sloupský potok a Bílá voda.")
samples = glob.glob(os.path.expanduser("~/voicebox/data/profiles/*/*.wav"))
ref = samples[0] if samples else None
print("Vzorek hlasu:", ref)

for repo in ["OpenMOSS-Team/MOSS-TTS-Local-Transformer",
             "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"]:
    print("Model:", repo)
    model = load(repo, lazy=True)
    parts = [r for r in model.generate(text=TEXT, ref_audio=ref, language="Czech", max_tokens=600)]
    audio = np.concatenate([np.array(r.audio) for r in parts], axis=0)
    out = os.path.expanduser(f"~/Desktop/moss_cs_{repo.split('/')[-1]}.wav")
    write(out, audio, parts[0].sample_rate)
    print("Ulozeno:", out)
    del model
