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
- Steering controls, from Zyphra's steering bundle (65 directions; `python3 -m rtx controls` lists them):
  `"controls": {"mood.valence": 2, "delivery.whisper": 1}` pushes a reply along emotion axes, speaking styles and
  word emphasis on any voice. `"speaker_mode": "embedding"` speaks a release voice by its speaker embedding, which
  the controls that reshape the voice itself need (`identity.age`, `texture.deep`, `accents_en.british`).
  `control_spans` steer a range of the text's characters, `steer_hold` and its two times set how the push eases
  off, and `controls_cfg_scale` is guidance on the controls (embedding voices). See "Steering".

- LACI: `"laci": true` watches where the model is in its text as it speaks and regenerates the part where it
  skipped words, got stuck or stopped early. A text of one or two words is spoken better by default: it is
  generated with a filler sentence after it and cut before the filler (`"short_text": "off"` turns that off).
  The text may hold pause markers (U+E003 to U+E006: a quarter, half, one and two seconds), a start marker
  (or `"start_immediately": true`: speech from the first frame) and pronunciation groups in IPA. See "LACI, short
  texts and markers".

Not served: steering spans placed by the model's reading position (`control_span_follow: "reading"`), a speaker
embedding made from a clip, LACI's duration targets, speech editing, accent
adapters, phoneme, start and pause tokens, and the release's `full_mr` decoder (the server uses `keep2_plain`, the
reference servers' default). `speed` and `instructions` are refused: the model has no such controls.

Arithmetic is fp32 throughout, checked against the publisher's code run in fp32 with exact gates. The publisher's
CUDA servers run the codec in bf16, so the codes this port encodes from a clip are not bit for bit the ones their
server encodes; the engine's page, `docs/recipes/zonos3-mini.md` in the pinned commit, measures the difference.

## Speed

One RTX 5090 (32 GB, driver 615.71, CUDA 13.3), the engine `python3 -m rtx build` makes, through the recipe harness's
`speech`, `speech-normalize`, `speech-controls`, `speech-concurrent`, `speech-steer` and `speech-laci` plans, the server
started for four requests at once. Each case was asked once, then three times whole and three times streamed. An idle second server
(this recipe's previous pin) held 7.8 GiB of the same card during the runs.

| Case | Audio | Whole reply | Times real time | Streamed: first byte, done |
|---|---:|---:|---:|---:|
| A long passage, voice `rupert` | 17.8 s | 0.48 s | 37.4 | 0.04 s, 0.48 s |
| A short sentence, the default voice | 4.13 s | 0.13 s | 32.3 | 0.03 s, 0.13 s |
| A second voice, `imogen` | 5.13 s | 0.15 s | 33.8 | 0.03 s, 0.16 s |
| A voice cloned from the short reply | 5.13 s | 0.72 to 0.73 s | 7.1 | 0.60 to 0.61 s, 0.72 to 0.73 s |
| A sentence, voice `rupert` (the `speech-controls` plan) | 5.40 s | 0.16 s | 33.3 | 0.03 s, 0.16 to 0.17 s |
| The same with speaker guidance 1.5 | 5.27 s | 0.18 s | 29.1 | 0.04 s, 0.18 s |
| The same at `speaking_rate` 8 | 6.40 s | 0.19 s | 34.2 | 0.03 to 0.04 s, 0.19 s |
| The same at `speaking_rate` 25 | 3.40 s | 0.11 s | 30.1 | 0.03 to 0.04 s, 0.12 s |
| The sentence steered: `mood.valence` 2, `delivery.whisper` 1 (the `speech-steer` plan) | 5.53 s | 0.17 s | 33.1 | 0.03 to 0.04 s, 0.17 s |
| The sentence by `rupert`'s speaker embedding | 5.20 s | 0.14 s | 36.6 | 0.02 s, 0.14 s |
| That voice reshaped: `identity.age` 1.5, `texture.deep` 1 | 4.87 s | 0.13 to 0.14 s | 36.3 | 0.02 s, 0.14 s |
| That voice with emphasis on characters 12 to 20 | 5.27 s | 0.15 s | 36.3 | 0.02 s, 0.15 s |
| That voice with `mood.arousal` 1 and controls guidance 1.5 | 5.00 s | 0.16 s | 31.1 | 0.03 s, 0.16 s |
| The long passage under LACI (the `speech-laci` plan) | 17.8 s | 0.48 s | 37.2 | 0.08 s, 0.48 s |
| "Hello." as a short text | 1.07 s | 0.06 s | 19.0 | 0.06 s, 0.06 s |
| "Hello." with `short_text` off | 0.87 s | 0.05 s | 17.0 | 0.03 s, 0.05 s |
| A sentence with a one-second pause marker | 2.73 s | 0.10 s | 28.2 | 0.03 s, 0.10 s |
| A sentence with `start_immediately` | 2.00 s | 0.08 s | 24.4 | 0.03 s, 0.08 s |
| A sentence with a pronunciation group | 3.20 s | 0.11 s | 29.9 | 0.03 to 0.04 s, 0.11 s |

- **Replies repeat.** Every case carries a seed; all four gave the same bytes every time, and the streamed bytes equal
  the whole reply's.
- **The audio decoder runs on cuBLAS**, NVIDIA's matrix library, since pin `c8c90565a`: the long passage took 0.61 s
  before. cuBLAS rounds a sum by how much it is given at once, so every reply is decoded in the same steps whether it
  is streamed or not, which keeps the two equal. Replies differ from the previous pin's by one step of 16-bit audio in
  about one sample in 2,000 (338 of 784,980 in the long passage, measured by hand), and another cuBLAS release may
  round differently again: replies are byte-stable on one install, not across installs.
- **Encoded replies:** the long passage as MP3 took 0.62 s, as Opus 0.53 s and as FLAC 0.49 s, each asked twice with
  equal bytes. AAC is not in the plan; it was checked by hand on the previous pin.
- **Text normalization** (the `speech-normalize` plan): six written sentences (three English, one each German,
  Portuguese and Japanese, from the normalizer's own test cases) each gave the same bytes as their expected spoken form
  sent with normalization off ("In 1990, he moved." as "In nineteen ninety, he moved."), and other bytes than the
  written text spoken as typed.
- **Condition values and guidance** (the `speech-controls` plan): raw values gave the bytes of the buckets they resolve
  to, all-unknown buckets and guidance 1 gave the bytes of a request without them, and guidance 1.5 gave other bytes.
  Guidance costs about a seventh more time here (29.1 against 33.3 times real time): the request's two passes a frame run
  as one. A negative guidance scale, an unknown condition name, a bucket past its table and a value that is not a number
  each got 400.
- **Cloning encodes the clip on every request**, which is the 0.6 s before a cloned reply's first byte. The
  first request with a clip also loads the clip encoder: 8.8 s here.
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
| The long passage, the short one, the guided one, the one with a rate | 32.2 s | 0.73 s | 44.4 |
| A clone, a guided clone, the long passage, the short one | 29.7 s | 1.83 s | 16.3 |
| The guided one, a clone, the one with a rate, a guided clone | 18.1 s | 1.55 s | 11.7 |
| The long passage four times | 71.2 s | 1.17 s | 60.8 |
| The long passage twice | 35.6 s | 0.70 s | 50.7 |

- **Requests at once share part of the work.** The requests in progress advance in one pass of the model a frame,
  so four copies of the long passage take 1.17 s, not four times the 0.48 s one takes alone. The audio decoder and
  the sampling are still done a request at a time.
- **A clip is encoded in one go**, about 0.6 s during which the other requests wait: the mixes with clones
  are the slow ones.
- **Six at once:** four were spoken, each the bytes it gives alone, and two got 503.

## Steering

`python3 -m rtx build` fetches Zyphra's steering bundle (MIT, 1.3 MB of direction vectors) from its code repository
at commit `96d9e0ec`, the folder `assets/steering` alone, and the server loads it; `steer = false` in `local.toml`
serves without it, and `controls` are then refused. `GET /v1/audio/controls` lists the 65 directions.

- **Two kinds of control.** Most work on any voice: they are added inside the model on every frame it generates
  (`mood.*`, `delivery.*`, `emotions.*`, `expressive.*`, `disfluency.*`, `emphasis_nv_nodur.emphasis`,
  `mandarin_hidden.cmn`). The others move the voice's speaker embedding (`identity.*`, `pitch.*`, `texture.*`,
  `accents_en.*`, `accents_l1.*`, `room.clean`, `mandarin.cmn`) and need `"speaker_mode": "embedding"`; on a voice
  by reference codes or a cloned clip they get 400.
- **A voice by its embedding is another rendering of the voice** than the default one by reference codes: the same
  speaker, other bytes, and here slightly faster (36.6 against 33.1 times real time; its prompt has one speaker row in place of the voice's reference frames).
  Cloning from a clip has no embedding mode: that needs the release's speaker encoder, which is not served.
- **Doses** are the publisher's raw doses, a multiple of the direction's vector; the runs here used 1 to 2. A dose
  of 0 gives the bytes of a request without the control.
- **Steer then hold.** By the bundle's default the push is full for the first 4 s of a reply, eases over 1 s to 0.35
  of the dose and stays there, which keeps a long reply from drifting; `delivery.tempo` and `delivery.whisper` stay
  constant. `"steer_hold": 1` keeps every control constant.
- **Checked** (the `speech-steer` plan): all six cases repeat byte for byte and stream the same bytes; a dose of 0,
  controls guidance 1, a span rate of 1 and a hold of 1 on an exempt control each gave the bytes of the request
  without them, and other bytes than their counterpart; an unknown control, a voice control on a reference voice, a
  span outside the text, both guidances at once, spans by reading position and a hold above 1 each got 400.
- **Cost:** a steered reply ran at 33.1 times real time, as the plain one did. Guidance on the controls runs the
  model twice a frame (31.1 against 36.6 for that voice plain).
- **What was not measured: whether a control sounds like its name.** The engine's page shows the port adds exactly
  what the publisher's code adds (identical frames in five test requests); no listening test or classifier was run
  on the results. Whisper transcribed five of the six replies with no errors; the whispered, steered one scored
  0.118 (two words of 17).

## LACI, short texts and markers

`python3 -m rtx build` also fetches the publisher's LACI alignment heads (`assets/laci_heads`, 43 files) and the
short-text registry of its Python package, from the same commit as the steering bundle; `laci = false` in
`local.toml` serves without them. The pause, start and phoneme tokens are in the release itself.

- **LACI** (`"laci": true`, `say --laci`): the server reads the model's place in the text after every frame. A
  skipped stretch, a stall or an early end rolls the reply back to where it went wrong and regenerates from
  there, up to eight times. It needs `speaker_cfg_scale` and `controls_cfg_scale` at 1.
  - On the long passage it changed nothing here and cost nothing: the same 17.8 s in 0.48 s. Streamed, the first
    byte came at 0.08 s where it otherwise comes at 0.04 s, because rows a rollback could still reach are held.
  - A streamed reply under LACI can differ from the whole one when a retry happens: rows already sent cannot be
    taken back, so the rollback starts later. Without a retry the two are the same bytes, as in all six cases here.
- **Short texts.** "Hello." alone took 0.87 s of audio; as a short text (the default) 1.07 s. The reply is held
  until its cut is decided, so it arrives in one piece even when streamed.
- **Markers** are private-use characters typed into the text: `\uE005` after a word is a pause of about a
  second, `\uE002` first is the start marker, and `\uE000...\uE001` holds IPA phones, spoken in place of a spelled word. An unknown marker, a start marker that is not first, an unclosed group and a phone outside
  the release's 136 get 400.
- **Checked** (the `speech-laci` plan): all six cases repeat byte for byte and stream the same bytes;
  `start_immediately` gave the bytes of the typed start marker and other bytes than without it; `laci: false`
  and `short_text: "auto"` gave the bytes of requests without them; six bad requests got 400.
- **Not measured:** how often LACI rescues a reply in practice, or whether short texts sound better this way.
  The engine's page shows the port does what the publisher's code does, retries included. Whisper transcribed
  five of the six replies without error; the one with a pronunciation group scored 0.2, where the reference text
  it is scored against holds the IPA and the markers.

## Memory

The engine's own count of device memory, at the recipe's four requests at once: 7,577,534,976 bytes after load, and
13,264,700,832 once a request with a clip has loaded the clip encoder and guided requests have made four second
lanes. Each request at once past the first costs 1,236,862,976 bytes, so `parallel = 1` loads in 3,866,946,048 and
`parallel = 8` in 12,524,986,880. cuBLAS's own allocations are outside that count; `nvidia-smi` showed the process
at 7,946 MiB after the load at four at once. The `nvidia-smi` lines in `evidence/` include the second server that sat on the
card. At four at once a card needs about 8 GiB free to speak and about 13.5 GiB to clone; at one, about 4.5 and 10.

`python3 -m rtx up` checks that the card has room for the load before it starts, and says so when there is not
enough left for cloning.

## The engine

The `zonos3` family is not in TensorFold's main yet: it is draft pull request
[ashhart/TensorFold#3720](https://github.com/ashhart/TensorFold/pull/3720). The recipe pins commit `0b3c39654`, kept as
the tag `zonos3-openai-0b3c39654` on github.com/BobClawblaw/TensorFold. That commit is the pull request's branch plus
eight changes not yet proposed upstream:

- MP3, Opus, AAC and FLAC replies and the sample rate on raw PCM (`docs/speech-formats.md` in the pinned commit);
- OpenAI's stock voice names (`alloy`, `nova`, ...) speaking in the default voice;
- text normalization (`docs/recipes/zonos3-mini.md` in the pinned commit, "Text normalization");
- condition values and speaker guidance (the same page, "Condition values and speaker guidance"): greedy frames
  equal to the publisher's code for both;
- several requests at once, advancing in one pass of the model (the same page, "Requests at once");
- the audio decoder's linear layers through cuBLAS (the same page, "The decoder's GEMM");
- steering controls (the same page, "Steering controls"): against the publisher's code with its bundle, the
  deltas and moved embeddings are bit-identical and the greedy frames identical in five requests;
- LACI, short texts and the phoneme, start and pause tokens (the same page, the sections of those names): the
  publisher's own replay fixtures pass, and on the model thirteen greedy requests, four of which retry eight times,
  give its readings, masks, rows and records.

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

Needs Linux with an NVIDIA card (run on an RTX 5090 only), CUDA 13 with `nvcc` and cuBLAS (`libcublas.so.13`, which
the decoder runs on), Zig 0.17.0, git, Python 3.11+ and
the `hf` command; Rust 1.88 or later for text normalization.

```sh
cp local.example.toml local.toml     # set zig, nvcc, cargo and where the release and a TensorFold clone live
python3 -m rtx fetch                 # the release at its pin, 15 GB, into model_dir
python3 -m rtx build                 # the pinned engine, the text normalizer, the steering bundle and the LACI heads, under ~/.cache/zonos3-mini
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
python3 -m rtx say "I can hardly believe it." -o glad.wav --voice rupert --control mood.valence=2
python3 -m rtx say "An older, deeper voice." -o older.wav --voice rupert --by-embedding --control identity.age=1.5 --control texture.deep=1
python3 -m rtx down
```

`say` picks the format from the file's extension (`.wav .mp3 .ogg .flac .aac .pcm`); `--language de` names the
normalizer's grammar and `--as-typed` turns normalization off; `--rate 10` and `--loudness -20` set condition values
and `--guidance 1.5` the speaker guidance scale; `--control NAME=DOSE` (repeatable) steers, `--by-embedding` speaks
the voice by its speaker embedding and `--controls-guidance 1.5` is guidance on the controls; `--laci` generates under
LACI, `--start` asks for speech from the first frame and `--no-short-text` speaks a word or two without the filler.
Other commands: `status`,
`logs`, `voices`, `controls` (the steering controls by kind), `command` (prints the serve command) and `systemd` (writes a systemd user unit that keeps
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
`speech-normalize`, `speech-controls`, `speech-concurrent`, `speech-steer` and `speech-laci` plans against the
`zonos3-mini-5090` target; the runs' files and the spoken replies are in `evidence/`.

```sh
git clone https://github.com/BobClawblaw/recipe-harness && cd recipe-harness
cp targets/zonos3-mini-5090.local.example.toml targets/zonos3-mini-5090.local.toml   # the engine, the release, asr
python3 -m harness run zonos3-mini-5090 --plan speech
python3 -m harness run zonos3-mini-5090 --plan speech-normalize
python3 -m harness run zonos3-mini-5090 --plan speech-controls
python3 -m harness run zonos3-mini-5090 --plan speech-concurrent
python3 -m harness run zonos3-mini-5090 --plan speech-steer
python3 -m harness run zonos3-mini-5090 --plan speech-laci
```

Each plan starts its own server, so `python3 -m rtx down` first. The word error rate needs `tools/asr.py`'s packages
(`torch`, `torchaudio`, `transformers`, `soundfile`) and downloads Whisper large-v3-turbo on first use; leave the
target's `asr` command out to skip it.

## License

The recipe is MIT. The model is Zyphra's, released under Apache-2.0, and its reference code under MIT. TensorFold is
Apache-2.0.
