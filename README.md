# ZONOS3 Mini on an RTX 5090 (TensorFold native, CUDA)

Zyphra's [ZONOS3 Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) text to speech, served from the files the publisher
released by [TensorFold](https://github.com/ashhart/TensorFold)'s native CUDA engine on one RTX 5090. It speaks
OpenAI's speech API (`POST /v1/audio/speech`), so a program written for OpenAI's text to speech, such as Hermes
Agent, uses it by changing its base URL. A browser test page comes up beside it.

- **Nothing is converted.** The server reads the release folder as downloaded, pinned to revision `d2d142a8`.
- **The engine is a branch under review**, not a TensorFold release (see "The engine").
- **Four requests at once** (`parallel` in `recipe.toml`, 1 to 16). They share the card's speed; a fifth gets 503.
- **No quality suite was run on this build.** The figures below are speed, repeatability and four transcribed
  sentences.

## What it serves

- The release's 27 voices (`/v1/voices`), a voice cloned from a WAV clip sent as `ref_audio`, or a voice given as
  reference codes (`speaker_reference_codes`).
- Replies as WAV (the default), MP3, Opus in Ogg, FLAC, AAC or raw PCM; `"stream": true` streams raw PCM.
- Numbers, dates and currency rewritten into words before they are spoken ("$5" is said "five dollars"), by the
  publisher's own normalizer. `language` picks its grammar (`en_us` by default, 44 codes such as `en_gb`, `fr_fr`,
  `de`, `ja`, `cmn`); `"text_normalization": false` speaks the text as typed.
- Speaking rate, loudness and bandwidth through the model's condition rows: `"condition_values": {"speaking_rate":
  10}` (UTF-8 bytes of text a second, about 6 to 30), `loudness` (LUFS) and `fmax` (Hz), or `condition_buckets` by
  bucket id. `speaker_cfg_scale` is speaker guidance: 1 is off, above 1 follows the voice harder, for
  about a tenth more time.
- Sampling fields `temperature`, `top_p`, `top_k` (1 to 4096) and `seed`. A seeded request gives the same bytes every
  time, in every format.
- Refusals with a message naming the problem: an unknown voice, a clip that is not a WAV file or is shorter than half a
  second, malformed reference codes, an empty input, `top_k` out of range, an unknown `language`, a condition name or
  bucket the model does not have, a negative guidance scale.

Not served: steering controls and their guidance, speech editing, accent
adapters, phoneme, start and pause tokens, and the release's `full_mr` decoder (the server uses `keep2_plain`, the
reference servers' default). `speed` and `instructions` are refused: the model has no such controls.

Arithmetic is fp32 throughout, checked against the publisher's code run in fp32 with exact gates. The publisher's
CUDA servers run the codec in bf16, so the codes this port encodes from a clip are not bit for bit the ones their
server encodes; the engine's page, `docs/recipes/zonos3-mini.md` in the pinned commit, measures the difference.

## Speed

One RTX 5090 (32 GB, driver 615.71, CUDA 13.3), the engine `python3 -m rtx build` makes, through the recipe harness's
`speech`, `speech-normalize`, `speech-controls` and `speech-concurrent` plans, the server started for four requests at
once. Each case was asked once, then three times whole and three times streamed. An idle second server (this recipe's
previous pin) held 4.3 GB of the same card during the runs.

| Case | Audio | Whole reply | Times real time | Streamed: first byte, done |
|---|---:|---:|---:|---:|
| A long passage, voice `rupert` | 17.8 s | 0.61 s | 29.1 | 0.04 s, 0.62 s |
| A short sentence, the default voice | 4.13 s | 0.16 s | 26.2 | 0.03 s, 0.16 s |
| A second voice, `imogen` | 5.13 s | 0.19 s | 26.9 | 0.04 s, 0.20 s |
| A voice cloned from the short reply | 5.13 s | 0.69 to 0.70 s | 7.4 | 0.54 s, 0.70 s |
| A sentence, voice `rupert` (the `speech-controls` plan) | 5.40 s | 0.20 s | 27.0 | 0.03 to 0.04 s, 0.20 to 0.21 s |
| The same with speaker guidance 1.5 | 5.27 s | 0.22 s | 23.9 | 0.04 s, 0.22 to 0.23 s |
| The same at `speaking_rate` 8 | 6.40 s | 0.23 to 0.24 s | 27.5 | 0.03 to 0.04 s, 0.24 s |
| The same at `speaking_rate` 25 | 3.40 s | 0.14 s | 25.2 | 0.04 s, 0.14 s |

- **Replies repeat.** Every case carries a seed; all four gave the same bytes every time, and the streamed bytes equal
  the whole reply's.
- **Encoded replies:** the long passage as MP3 took 0.76 s, as Opus 0.66 s and as FLAC 0.62 s, each asked twice with
  equal bytes. AAC is not in the plan; it was checked by hand on the previous pin.
- **Text normalization** (the `speech-normalize` plan): six written sentences (three English, one each German,
  Portuguese and Japanese, from the normalizer's own test cases) each gave the same bytes as their expected spoken form
  sent with normalization off ("In 1990, he moved." as "In nineteen ninety, he moved."), and other bytes than the
  written text spoken as typed.
- **Condition values and guidance** (the `speech-controls` plan): raw values gave the bytes of the buckets they resolve
  to, all-unknown buckets and guidance 1 gave the bytes of a request without them, and guidance 1.5 gave other bytes.
  Guidance costs about a tenth more time here (23.9 against 27.0 times real time): the request's two passes a frame run
  as one. A negative guidance scale, an unknown condition name, a bucket past its table and a value that is not a number
  each got 400.
- **Cloning encodes the clip on every request**, which is the half second before a cloned reply's first byte. The
  first request with a clip also loads the clip encoder: 8.9 s here.
- **Three requests sent together** were all spoken, each the bytes the request gives alone.
- **Refusals:** an unknown voice, `top_k` above 4096, a `ref_audio` that is not a WAV file, malformed
  `speaker_reference_codes` and an empty input each got 400, and so did an unknown `language` in the second plan.
- **Intelligibility:** Whisper large-v3-turbo (`tools/asr.py`) transcribed the four replies of the `speech` plan with
  word error rate 0, and so the plain, guided and slow replies of `speech-controls`; the reply at `speaking_rate` 25
  scored 0.059 (one word of 17), and in `speech-concurrent` the sentence with "$5" and "3 days" at `speaking_rate` 9
  scored 0.111 (one word of nine) while its other five replies scored 0. That is a handful of English sentences, not a
  quality evaluation: no listening test, and no comparison of voices against the publisher's own server.

### Several at once

The `speech-concurrent` plan asks six kinds of request alone (the long passage, a short one, a guided one, one with a
speaking rate and normalized text, a cloned voice, a guided clone), then sends mixes of them together. All 18 replies
in the five mixes equalled the same request's reply alone, byte for byte.

| Sent together | Audio in all | Until the last reply | Times real time in all |
|---|---:|---:|---:|
| The long passage, the short one, the guided one, the one with a rate | 32.2 s | 0.96 s | 33.4 |
| A clone, a guided clone, the long passage, the short one | 29.7 s | 1.93 s | 15.4 |
| The guided one, a clone, the one with a rate, a guided clone | 18.1 s | 1.54 s | 11.7 |
| The long passage four times | 71.2 s | 1.71 s | 41.7 |
| The long passage twice | 35.6 s | 0.98 s | 36.4 |

- **Requests at once share part of the work.** The requests in progress advance in one pass of the model a frame,
  so four copies of the long passage take 1.71 s, not four times the 0.61 s one takes alone. The audio decoder and
  the sampling are still done a request at a time, and the decoder is now the larger part.
- **A clip is encoded in one go**, about half a second during which the other requests wait: the mixes with clones
  are the slow ones.
- **Six at once:** four were spoken, each the bytes it gives alone, and two got 503.

## Memory

The engine's own count of device memory, at the recipe's four requests at once: 7,538,995,456 bytes after load, and
13,226,161,312 once a request with a clip has loaded the clip encoder and guided requests have made their second
lanes. Each request at once past the first costs 1,236,862,976 bytes, so `parallel = 1` loads in 3,828,406,528 and
`parallel = 8` in 12,486,447,360. The `nvidia-smi` lines in `evidence/` include the second server that sat on the
card. At four at once a card needs about 7.5 GiB free to speak and about 13 GiB to clone; at one, about 4.5 and 10.5.

`python3 -m rtx up` checks that the card has room for the load before it starts, and says so when there is not
enough left for cloning.

## The engine

The `zonos3` family is not in TensorFold's main yet: it is draft pull request
[ashhart/TensorFold#3720](https://github.com/ashhart/TensorFold/pull/3720). The recipe pins commit `a158fcdaf`, kept as
the tag `zonos3-openai-a158fcdaf` on github.com/BobClawblaw/TensorFold. That commit is the pull request's branch plus
five changes not yet proposed upstream:

- MP3, Opus, AAC and FLAC replies and the sample rate on raw PCM (`docs/speech-formats.md` in the pinned commit);
- OpenAI's stock voice names (`alloy`, `nova`, ...) speaking in the default voice;
- text normalization (`docs/recipes/zonos3-mini.md` in the pinned commit, "Text normalization");
- condition values and speaker guidance (the same page, "Condition values and speaker guidance"): greedy frames
  equal to the publisher's code for both;
- several requests at once, advancing in one pass of the model (the same page, "Requests at once").

In the pull request: the backbone, the codec decoder and the clip encoder as 40 CUDA kernels, each registered with
the kernel contract (`41 kernels: 41 pass gates 1-4, 0 fail, 0 known findings; 41 controls: 41 red as planted, 0
missed`), the release's `.pth` files read without Python, and memory admission for the load and the clip encoder.

MP3, Opus and AAC come from libraries the server loads when a reply is first asked in that format: LAME, libopus and
FDK AAC (Debian or Ubuntu: `sudo apt install libmp3lame0 libopus0 libfdk-aac2`). A format whose library is missing
is refused with the line to act on; WAV, PCM and FLAC need nothing. `validate` and `up` say which are missing.

Text normalization is Zyphra's [zyphra-textnorm](https://github.com/Zyphra/zyphra-textnorm) 0.2.0 (MIT, with
Apache-2.0 grammars from NVIDIA NeMo), the Rust library the publisher's own servers use. `python3 -m rtx build`
compiles it at commit `170c571e` and puts it in `lib/` beside the engine, where the server loads it on the first
request. That needs Rust 1.88 or later (`cargo` in `local.toml`). Without it the build says so and everything else
works: text is then spoken as typed, and `"text_normalization": true` is refused.

## Use

Needs Linux with an NVIDIA card (run on an RTX 5090 only), CUDA 13 with `nvcc`, Zig 0.17.0, git, Python 3.11+ and
the `hf` command; Rust 1.88 or later for text normalization.

```sh
cp local.example.toml local.toml     # set zig, nvcc, cargo and where the release and a TensorFold clone live
python3 -m rtx fetch                 # the release at its pin, 15 GB, into model_dir
python3 -m rtx build                 # the pinned engine and the text normalizer, under ~/.cache/zonos3-mini
python3 -m rtx validate
python3 -m rtx up --foreground       # the API on :30117 and the test page on :30118; Ctrl-C stops both
```

Then open `http://127.0.0.1:30118/`: the page lists the voices, speaks what you type with a seed and the sampling
fields, keeps each take with a player and its timing, and clones a voice from an audio file you pick.

Without `--foreground`, `up` starts both in the background and returns:

```sh
python3 -m rtx up
python3 -m rtx say "The harbor was quiet this morning." -o reply.mp3 --voice rupert
python3 -m rtx say "Now in the clip's voice." -o cloned.wav --ref clip.wav
python3 -m rtx down
```

`say` picks the format from the file's extension (`.wav .mp3 .ogg .flac .aac .pcm`); `--language de` names the
normalizer's grammar and `--as-typed` turns normalization off; `--rate 10` and `--loudness -20` set condition values
and `--guidance 1.5` the speaker guidance scale. Other commands: `status`,
`logs`, `voices`, `command` (prints the serve command) and `systemd` (writes a systemd user unit that keeps
`up --foreground` running; enabling it is left to you). Settings live in `recipe.toml`; `local.toml` and upper-case
environment variables override them (`PORT=30200 python3 -m rtx up`). `parallel` is how many requests are spoken at
once (4 here; each one past the first takes 1.2 GiB more of the card). To serve other machines set `host = "0.0.0.0"`
and an `api_key`. The test page sends no key, so with an `api_key` it lists nothing and cannot speak: set
`page_port = 0` then. `page_bind = "0.0.0.0"` opens the page to other machines, with no password.

A request by hand:

```sh
curl -s http://127.0.0.1:30117/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model": "zonos3-mini", "input": "The harbor was quiet this morning.", "voice": "rupert", "seed": 1,
       "response_format": "mp3"}' -o reply.mp3
```

Clone only voices you have the right to use.

### From Hermes and other OpenAI speech clients

For [Hermes Agent](https://github.com/NousResearch/hermes-agent), in `~/.hermes/config.yaml`:

```yaml
tts:
  provider: openai
  openai:
    api_key: local            # any value, unless you set api_key in the recipe
    base_url: http://127.0.0.1:30117/v1
    model: zonos3-mini
    voice: rupert             # optional: any name from /v1/voices
```

Checked with Hermes's own code against this server: its speech tool wrote MP3 and Ogg Opus files of the text it was
given, also with no `voice` set (Hermes then sends `alloy`, which speaks in the default voice), and its streaming
client received raw PCM and took the 44,100 Hz rate from the reply's header. Run as a whole agent from its command
line, Hermes called the speech tool itself and wrote an MP3 of the sentence it was asked to say. A voice reply to a
message arriving through a chat platform was not checked.

The [Hermes web UI](https://github.com/nesquena/hermes-webui) reads the same `tts.openai` settings, but its OpenAI
speech engine will not call any server until a key is in its environment. Put a placeholder in the web UI's `.env`
(`VOICE_TOOLS_OPENAI_KEY=local`), restart it, and pick Settings → Voice → TTS Engine → OpenAI. It accepts a plain
`http` base URL only for `localhost` or `127.0.0.1`, so the web UI has to run on the machine that serves the model.

OpenAI's `speed` is refused; the model's own control is `condition_values.speaking_rate`, which Hermes does not send.

What differs from OpenAI's own service: a request without `response_format` gets WAV, not MP3; raw PCM is 44,100 Hz,
not 24,000, and says so in the reply's `X-Audio-Sample-Rate` header; `speed` and `instructions` are refused; a request
past the server's `parallel` gets 503.

## Checking it again

The figures come from the [recipe harness](https://github.com/BobClawblaw/recipe-harness)'s `speech`,
`speech-normalize`, `speech-controls` and `speech-concurrent` plans against the `zonos3-mini-5090` target; the runs'
files and the spoken replies are in `evidence/`.

```sh
git clone https://github.com/BobClawblaw/recipe-harness && cd recipe-harness
cp targets/zonos3-mini-5090.local.example.toml targets/zonos3-mini-5090.local.toml   # the engine, the release, asr
python3 -m harness run zonos3-mini-5090 --plan speech
python3 -m harness run zonos3-mini-5090 --plan speech-normalize
python3 -m harness run zonos3-mini-5090 --plan speech-controls
python3 -m harness run zonos3-mini-5090 --plan speech-concurrent
```

Each plan starts its own server, so `python3 -m rtx down` first. The word error rate needs `tools/asr.py`'s packages
(`torch`, `torchaudio`, `transformers`, `soundfile`) and downloads Whisper large-v3-turbo on first use; leave the
target's `asr` command out to skip it.

## License

The recipe is MIT. The model is Zyphra's, released under Apache-2.0, and its reference code under MIT. TensorFold is
Apache-2.0.
