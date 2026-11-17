#!/usr/bin/env python3
"""A cloning request for /v1/audio/speech as JSON: clone_request.py CLIP.wav "text to speak" [seed]"""
import base64
import json
import sys

clip, text = sys.argv[1], sys.argv[2]
print(json.dumps({"model": "zonos3-mini", "input": text, "seed": int(sys.argv[3]) if len(sys.argv) > 3 else 1,
                  "ref_audio": base64.b64encode(open(clip, "rb").read()).decode()}))
