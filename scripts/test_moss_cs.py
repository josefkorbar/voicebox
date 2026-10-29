# Rychly test cestiny v MOSS-TTS (mimo Voicebox). Spusteni:
#   uv run --python 3.12 --with mlx-audio==0.6.3 python test_moss_cs.py [model ...]
# Bez argumentu zkusi 1.7B a v1.5 (4B). Velky 8B model: OpenMOSS-Team/MOSS-TTS-v1.5
# Kazdy model zkusi dva rezimy klonovani:
#   generation   = jen vzorek hlasu
#   continuation = vzorek hlasu + jeho prepis (obvykle vernejsi hlas)
import os, sqlite3, sys
import numpy as np
from mlx_audio.tts import load
from mlx_audio.audio_io import write, read

TEXT = ("Jeskynní potápěči objevili v Amatérské jeskyni v Moravském krasu místo, "
        "kde se spojuje Sloupský potok a Bílá voda.")

data = os.path.expanduser("~/voicebox/data")
db = sqlite3.connect(os.path.join(data, "voicebox.db"))
audio_path, ref_text = db.execute(
    "select audio_path, reference_text from profile_samples order by rowid desc limit 1").fetchone()
ref = audio_path if os.path.isabs(audio_path) else os.path.join(data, audio_path)
arr, sr = read(ref)
print(f"Vzorek hlasu: {ref} ({len(arr) / sr:.1f} s)")
print("Prepis vzorku:", ref_text)

for repo in sys.argv[1:] or ["OpenMOSS-Team/MOSS-TTS-Local-Transformer",
                            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"]:
    model = load(repo, lazy=True)
    for mode in ["generation", "continuation"]:
        print("Model:", repo, "| rezim:", mode)
        parts = list(model.generate(text=TEXT, ref_audio=ref, ref_text=ref_text,
                                    mode=mode, language="Czech", max_tokens=600))
        audio = np.concatenate([np.array(r.audio) for r in parts], axis=0)
        out = os.path.expanduser(f"~/Desktop/moss_cs_{repo.split('/')[-1]}_{mode}.wav")
        write(out, audio, parts[0].sample_rate)
        print("Ulozeno:", out)
    del model
