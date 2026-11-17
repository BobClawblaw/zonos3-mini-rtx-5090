# ZONOS3 Mini text to speech on one RTX 5090

Zyphra's [ZONOS3 Mini](https://huggingface.co/Zyphra/ZONOS3-Mini), served from the files the publisher released, on
[TensorFold](https://github.com/ashhart/TensorFold)'s native CUDA engine: `POST /v1/audio/speech` with the release's
27 voices, a voice cloned from a clip you upload, WAV or raw PCM, whole or streamed.

- **Nothing is converted.** The server reads the release folder as downloaded, pinned to revision `d2d142a8`.
- **The engine is a branch under review**, not a TensorFold release: draft pull request
  [ashhart/TensorFold#3720](https://github.com/ashhart/TensorFold/pull/3720). This recipe pins one commit of it.
- **One request at a time.** A request that arrives while another is being spoken gets 503.

## Measured on this card

One RTX 5090 (32 GB, driver 615.71, CUDA 13.3), the build `setup.sh` makes, through the
[recipe harness](https://github.com/BobClawblaw/recipe-harness)'s `speech` plan (target `zonos3-mini-5090`). The
run's files and the spoken replies are in `evidence/2026-11-17-speech/`.

| Case | Audio | Whole reply | Times real time | Streamed: first byte, done |
|---|---:|---:|---:|---:|
| A long passage, voice `rupert` | 17.8 s | 0.61 s | 29.4 | 0.04 s, 0.62 s |
| A short sentence, the default voice | 4.13 s | 0.16 s | 26.2 | 0.03 s, 0.16 s |
| A second voice, `imogen` | 5.13 s | 0.19 s | 27.0 | 0.04 s, 0.20 s |
| A voice cloned from the short reply | 5.13 s | 0.69 to 0.70 s | 7.4 | 0.54 to 0.55 s, 0.70 s |

Each case was asked once, then three times whole and three times streamed.

- **Replies repeat.** Every request carries a seed; all four cases gave the same bytes every time, and the streamed
  bytes equal the whole reply's.
- **Cloning encodes the clip on every request**, which is the half second before a cloned reply's first byte. The
  first request that brings a clip also loads the clip encoder: it took 13.7 s here.
- **A busy server refuses.** Of three requests sent together, one was spoken (the same bytes as alone) and two got 503.
- **Refusals:** an unknown voice, `top_k` above 4096, a `ref_audio` that is not a WAV file, malformed
  `speaker_reference_codes` and an empty input each got 400 with a message naming the problem.
- **Device memory** (the engine's own count): 3,828,406,528 bytes after load, 9,471,917,216 once the clip encoder is
  loaded. `nvidia-smi` showed 4,282 MiB and 10,092 MiB.
- **Intelligibility:** Whisper large-v3-turbo (`tools/asr.py`) transcribed all four replies with word error rate 0.
  That is four sentences in English, not a quality evaluation: no quality suite and no listening test was run on
  this build, and no comparison of voices against the publisher's own server.

## What it does not do

Said plainly, because the release supports more than this port serves:

- No concurrency, and no Metal build: CUDA only, one request at a time.
- Text is spoken as typed: no text normalization, so write numbers and abbreviations the way they should be said.
- Reference clips must be WAV files (PCM of 8, 16, 24 or 32 bits, or 32-bit float), at least half a second long.
- Not ported: speaker and control guidance, condition values, speech editing, accent adapters, phoneme, start and
  pause tokens, and the `full_mr` decoder (the server uses `keep2_plain`, the reference servers' default).
- Arithmetic is fp32 throughout, checked against the publisher's code run in fp32. The publisher's CUDA servers run
  the codec in bf16, so the codes this port encodes from a clip are not bit-for-bit the ones their server encodes.
  The engine's page, `docs/recipes/zonos3-mini.md` in the pinned branch, measures the difference.

## Setup

Needs an NVIDIA card with about 11 GB free (measured on a 5090 only), CUDA 13 with `nvcc`, Zig 0.17.0, the `hf`
command, and about 15 GB of disk for the release.

    ./setup.sh        # downloads the release at its pin, checks out the engine at its pin, builds it for sm_120
    ./run.sh          # the server and its test page, in this terminal; Ctrl-C stops both

Then open `http://127.0.0.1:30118/`: the test page lists the voices, speaks what you type with a seed and the
sampling fields, keeps each take with a player and its timing, and clones a voice from an audio file you pick. The
API is on `http://127.0.0.1:30117/v1`.

- **From another machine:** `UI_BIND=0.0.0.0 ./run.sh` lets other machines open the page on port 30118. The page has
  no password, so do that only on a network you trust.
- **Other ports:** `PORT=... UI_PORT=... ./run.sh`.
- **The API alone:** `./serve.sh` (its header lists `PORT`, `HOST`, `NAME`).
- **As a service, if you want one:** `./service.sh install` keeps `run.sh` running as a systemd user service (no
  root; it starts at login and restarts on failure), and `./service.sh remove` takes it out again.

`recipe.toml` holds the pins. For another card, change `sm` in `recipe.toml`; only sm_120 was run. The page ships
with the engine (`tools/zig/zonos3_test_page.py`, standard library only). It was loaded in headless Chromium, where
it listed the voices without a script error, and its routes were exercised with curl; its buttons were not clicked
by an automated test.

## Speaking

The same server from the command line. A voice from the release (`curl -s 127.0.0.1:30117/v1/voices` lists them):

    curl -s 127.0.0.1:30117/v1/audio/speech -H 'Content-Type: application/json' \
      -d '{"model": "zonos3-mini", "input": "The harbor was quiet this morning.", "voice": "rupert", "seed": 1}' -o reply.wav

Streamed raw PCM (16-bit mono, 44.1 kHz), written as it arrives:

    curl -sN 127.0.0.1:30117/v1/audio/speech -H 'Content-Type: application/json' \
      -d '{"model": "zonos3-mini", "input": "Streaming starts at once.", "response_format": "pcm", "stream": true, "seed": 1}' -o reply.pcm

A voice cloned from your own clip (a WAV file), sent as base64 in `ref_audio`:

    tools/clone_request.py clip.wav "Now in the clip's voice." > request.json
    curl -s 127.0.0.1:30117/v1/audio/speech -H 'Content-Type: application/json' -d @request.json -o cloned.wav

Clone only voices you have the right to use. Sampling fields are `temperature`, `top_p`, `top_k` (1 to 4096) and
`seed`; send a seed when a reply has to repeat.

## Measuring it yourself

    git clone https://github.com/BobClawblaw/recipe-harness && cd recipe-harness
    cp targets/zonos3-mini-5090.local.example.toml targets/zonos3-mini-5090.local.toml   # fill in the paths
    python3 -m harness run zonos3-mini-5090 --plan speech

The plan starts the server on port 30117, runs the cases above and leaves it running. The word error rate needs
`tools/asr.py`'s packages (`torch`, `torchaudio`, `transformers`, `soundfile`) and downloads Whisper on first use;
leave the target's `asr` command out to skip it.

## License

This recipe's scripts are MIT (see `LICENSE`). The ZONOS3 Mini release is Apache-2.0 and its reference code MIT, both
Zyphra's; TensorFold has its own license.
