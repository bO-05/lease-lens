# Codex Build Log

This log documents the Codex-assisted final submission pass for Lease Lens.

## Goal

Prepare Lease Lens for the Build Small Hackathon with maximum near-deadline leverage:

- make the Space easier for judges to evaluate quickly;
- make the OpenAI Codex Track evidence explicit;
- strengthen machine-readable category tags;
- preserve the existing small-model, offline, and measured-evaluation claims.

## Codex-Assisted Changes

- Added canonical hackathon tags to the Space README:
  `track:backyard`, `sponsor:openai`, `sponsor:modal`, `achievement:offgrid`,
  `achievement:welltuned`, `achievement:offbrand`, `achievement:llama`, and
  `achievement:fieldnotes`.
- Added an OpenAI Codex Track section to the README with Codex attribution
  evidence and a short judge path.
- Added this build log as the public provenance artifact for reviewers once the
  repo is published.
- Replaced the visible stock Gradio Blocks interface with a custom redline legal
  evidence desk served through `gradio.Server`, with a styled Blocks fallback.
- Added public frontend source files: `index.html`, `static/app.css`,
  `static/app.js`, and `static/lease-lens-mark.svg`.
- Added REST fallback routes beside the Gradio client APIs so the custom
  frontend still works if a browser blocks the CDN JS client.
- Updated the Gradio app to default to a real SEC-filed lease and show the SEC
  provenance banner on first load.
- Kept model loading, prompting, scoring, extraction guards, and generation
  behavior unchanged.

## Verification Commands

Run these before publishing:

```bash
python -m py_compile app.py sample_contracts.py
python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text(encoding='utf-8')); ast.parse(pathlib.Path('sample_contracts.py').read_text(encoding='utf-8')); print('AST checks passed')"
```

After pushing to Hugging Face, verify the Space metadata:

```bash
python -c "import json, urllib.request; d=json.load(urllib.request.urlopen('https://huggingface.co/api/spaces/build-small-hackathon/lease-lens')); print(d['runtime']['stage'], d['runtime']['hardware']['current'], d['private'], d['sha'])"
```

Expected:

- runtime stage is `RUNNING`;
- hardware is `zero-a10g`;
- private is `False`;
- SHA matches the latest pushed Space commit.

## Local Codex-Attributed Commits

- `df9f20d35448693d6307dc8be275963e4b90fbc5` - prepared the hackathon submission package, README proof, and app default.
- `3a974831a08b2fd34448ee8fc7b58b9e8448c470` - recorded the first Codex provenance commit in this build log.
- `aa802c421f00a665c392a7beb1114e11c6c3592c` - added the custom legal evidence desk UI, `gradio.Server` APIs, mock preview mode, and REST fallback routes.

## Limitations Kept Honest

- Lease Lens is a review assistant, not legal advice.
- The shipped app relies on deterministic grounding, deduplication, and keyword
  relevance guards around a 3B fine-tune.
- The app reads the first 80k characters of long documents and declares coverage
  in the UI.
- The v2/v2.5 abstention experiments are linked as measured variants rather than
  silently replacing the shipped model.
