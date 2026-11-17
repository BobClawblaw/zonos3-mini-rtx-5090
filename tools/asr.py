"""Transcribe audio files with Whisper large-v3-turbo and print the word error rate against a text: asr.py TEXT file..."""
import sys, re, numpy as np, torch, torchaudio, soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
def words(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)
name = "openai/whisper-large-v3-turbo"
proc = WhisperProcessor.from_pretrained(name)
model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
ref = words(sys.argv[1])
for path in sys.argv[2:]:
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(pcm.mean(1)), rate, 16000)
    feats = proc(x.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
    ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=400)
    out = proc.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"{path}: {len(x)/16000:.1f}s WER {wer(ref, words(out)):.3f} | {out.strip()}")
