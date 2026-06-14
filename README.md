---
title: Lease Lens
emoji: 🔍
colorFrom: indigo
colorTo: blue
pinned: true
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
license: apache-2.0
short_description: A 3B legal model that reads the lease before you sign it.
tags:
- gradio
- build-small-hackathon
- track:backyard
- sponsor:openai
- sponsor:modal
- achievement:offgrid
- achievement:welltuned
- achievement:offbrand
- achievement:llama
- achievement:fieldnotes
- backyard ai
- backyard-ai
- codex
- openai
- best use of codex
- best-use-of-codex
- tiny titan
- tiny-titan
- well tuned
- well-tuned
- off brand
- off-brand
- llama champion
- llama-champion
- llama.cpp
- off the grid
- off-the-grid
- modal
- best use of modal
- best-use-of-modal
- field notes
- field-notes
- best demo
- best-demo
- community choice
- community-choice
- bonus quest champion
- bonus-quest-champion
- judges wildcard
- judges-wildcard
- zerogpu
- legal
- contracts
models:
- giladam01/lease-lens-legal-3b
- giladam01/lease-lens-legal-3b-gguf
- giladam01/lease-lens-legal-3b-v2
- giladam01/lease-lens-legal-3b-v25
datasets:
- chenghao/cuad_qa
---

<div align="center">

# 🔍 Lease Lens

**A 3-billion-parameter legal model that reads the lease before you sign it.**

*Made by [giladam01](https://huggingface.co/giladam01)*

[![Live Space](https://img.shields.io/badge/🤗_Live-Space-FFD21E?style=flat-square)](https://huggingface.co/spaces/build-small-hackathon/lease-lens)
[![Model](https://img.shields.io/badge/🤗_Model-lease--lens--legal--3b-FF9D00?style=flat-square)](https://huggingface.co/giladam01/lease-lens-legal-3b)
[![GGUF](https://img.shields.io/badge/llama.cpp-GGUF-06B6D4?style=flat-square)](https://huggingface.co/giladam01/lease-lens-legal-3b-gguf)
[![Field Notes](https://img.shields.io/badge/Read-Field_Notes-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/blog/giladam01/lease-lens-article)
[![GitHub Repo](https://img.shields.io/badge/GitHub-bO--05%2Flease--lens-111827?style=flat-square&logo=github&logoColor=white)](https://github.com/bO-05/lease-lens)
[![Codex Log](https://img.shields.io/badge/Codex-build_log-111827?style=flat-square&logo=openai&logoColor=white)](docs/codex-build-log.md)

</div>

Paste any contract → verbatim risky-clause flags, a risk score, in-text highlighting, plain-English "push back" tips, and a one-click negotiation email. The entire model runs inside this Space — **no external LLM API is ever called**.

The Space now uses a custom **redline legal evidence desk** frontend around the same Gradio/ZeroGPU backend: a real SEC filing loads by default, the judge path is visible on the first screen, and results render as a risk docket with clause evidence and a negotiation letter panel.

## Submission Snapshot

| | |
|---|---|
| **Live Space** | [build-small-hackathon/lease-lens](https://huggingface.co/spaces/build-small-hackathon/lease-lens) |
| **Public GitHub repo** | [bO-05/lease-lens](https://github.com/bO-05/lease-lens) |
| **Field Notes article** | [What fine-tuning a 3B legal model taught me](https://huggingface.co/blog/giladam01/lease-lens-article) |
| **Track** | Backyard AI — a contract-defense tool for people who sign things they don't read |
| **Model (shipped)** | [`giladam01/lease-lens-legal-3b`](https://huggingface.co/giladam01/lease-lens-legal-3b) — Llama-3.2-3B fine-tune (≤4B) |
| **Local runtime** | [GGUF](https://huggingface.co/giladam01/lease-lens-legal-3b-gguf) for llama.cpp / Ollama |
| **Training data** | [CUAD](https://huggingface.co/datasets/chenghao/cuad_qa) (CC-BY-4.0) |

## TL;DR for Judges

- **OpenAI Codex Track:** Lease Lens was prepared with Codex as the coding agent; see [`docs/codex-build-log.md`](docs/codex-build-log.md) for the public build evidence.
- **Backyard AI:** a practical contract-defense tool for the people who sign leases, NDAs, and gym contracts without reading them — try it on the **three real SEC-filed leases built into the app**.
- **Tiny Titan / Well-Tuned:** `Llama-3.2-3B` + [our CUAD fine-tune](https://huggingface.co/giladam01/lease-lens-legal-3b) — **+242% relative F1 over base (0.119→0.406), exact-match 28×, paired-bootstrap 95% CI [+0.211,+0.368]**, on 100 held-out CUAD items with **pre-registered pass/fail gates**.
- **Llama Champion:** the same fine-tune ships as a [Q8_0 GGUF](https://huggingface.co/giladam01/lease-lens-legal-3b-gguf) for the llama.cpp runtime — `ollama pull hf.co/giladam01/lease-lens-legal-3b-gguf` runs it fully offline.
- **Best Use of Modal:** the [v2.5 variant](https://huggingface.co/giladam01/lease-lens-legal-3b-v25) (20% synthesized-abstention mix) was trained end-to-end on a **Modal A100** — see [`training/finetune_legal_3b_modal_v2.py`](training/finetune_legal_3b_modal_v2.py) and the smoke-run evidence in [`docs/modal-evidence.md`](docs/modal-evidence.md).
- **Off-Brand / Off the Grid:** custom-branded dark UI (beyond stock Gradio), and **zero cloud-API architecture** — weights run in-process; the GGUF runs on a laptop with no internet.
- **Field Notes / Bonus Quest:** we published our failures too — see the field notes: the model that *never said NONE* (100% FP on absent clauses) and the negatives retrain that fixed it (→4% FP) at a recall cost, all measured.
- **Submission package:** [Live Space](https://huggingface.co/spaces/build-small-hackathon/lease-lens) · [model](https://huggingface.co/giladam01/lease-lens-legal-3b) · [GGUF](https://huggingface.co/giladam01/lease-lens-legal-3b-gguf) · [v2 abstention variant](https://huggingface.co/giladam01/lease-lens-legal-3b-v2) · [v2.5 Modal-trained](https://huggingface.co/giladam01/lease-lens-legal-3b-v25)

- **GitHub repo:** [bO-05/lease-lens](https://github.com/bO-05/lease-lens) with Codex-attributed commits for the OpenAI Codex Track.

## OpenAI Codex Track

Lease Lens is being finalized with Codex as the coding agent for the OpenAI Codex Track.

- **Public GitHub repo:** [`bO-05/lease-lens`](https://github.com/bO-05/lease-lens).
- **Codex-attributed commits:** local commits use a `Co-authored-by: OpenAI Codex <codex@openai.com>` trailer so the public history makes the Codex contribution explicit.
- **Build provenance:** [`docs/codex-build-log.md`](docs/codex-build-log.md) records what Codex changed, which checks were run, and what model/runtime behavior was intentionally left unchanged.
- **Judge path:** open the Space, keep the default real SEC lease selected, press **Analyze contract**, then draft the negotiation email from the flags.

## Proven, not vibes

Same 100 held-out CUAD extraction items, all models, seed-fixed, paired bootstrap:

| model | F1 | Exact match |
|---|---|---|
| Llama-3.2-3B (base) | 0.119 | 0.010 |
| **Lease Lens 3B (this app)** | **0.406** | **0.280** |
| Llama-3.1-8B (base) | 0.206 | 0.020 |
| our 8B fine-tune | 0.357 | 0.230 |

The 3B fine-tune **beats our own 8B fine-tune** on identical items — small, tuned right, wins. ΔF1 95% CI **[+0.211, +0.368]** (excludes zero). OOD check (ContractNLI, n=60): parity with base → no catastrophic forgetting.

**Honest limitation, openly measured:** trained on positives only, the bare model over-extracts on absent clause types (100% FP on a synthesized no-answer benchmark). The app contains this with three deterministic guards (verbatim grounding, dedup, keyword relevance). A negatives-trained variant ([v2](https://huggingface.co/giladam01/lease-lens-legal-3b-v2)) cuts FP to **4%** at an extraction-recall cost (F1 0.300) — we shipped v1 + guards and published both, plus a Modal-trained 20%-mix [v2.5](https://huggingface.co/giladam01/lease-lens-legal-3b-v25).

## Tested on real contracts (not just benchmarks)

Three genuine executed leases from SEC EDGAR filings (2024-2025 — provably outside CUAD):

Each is built into the app's dropdown and links to its original SEC filing — click through and verify it yourself:

| real lease (SEC source) | filer | result |
|---|---|---|
| [Office lease — Alpharetta, GA](https://www.sec.gov/Archives/edgar/data/1879403/000121390024033444/ea020177001ex10-131_larosa.htm) (EX-10.131) | La Rosa Holdings Corp. | **56/100 · 6 verbatim flags** in the latest live smoke scan |
| [Office lease amendment — Boston, MA](https://www.sec.gov/Archives/edgar/data/2023658/000202365825000098/exhibit102-116huntingtonxb.htm) (EX-10.2, 2025) | Bicara Therapeutics Inc. | **31/100 · 3 flags** — incl. the exact **$125,301.33** security-deposit clause |
| [Office lease — Addison, TX](https://www.sec.gov/Archives/edgar/data/1494259/000095017024020157/carg-ex10_31.htm) (EX-10.31) | CarGurus / CarOffer LLC | long-form coverage stress test with the app's live score, grounded flags, and coverage note |

*These are genuine executed commercial leases filed as public exhibits with the U.S. SEC (2024-2025) — outside the CUAD training set. Lease Lens has never seen them in training.*

## How it works

- **One batched pass on ZeroGPU**: the contract is split into overlapping 5k windows (first 80k chars); each of 10 clause categories is routed only to windows containing its keywords; all checks run as a single batched greedy generate → **seconds per analysis**.
- **Three guards** keep flags honest: the quote must appear verbatim in the contract, can't repeat across categories, and must contain category-relevant terms.
- **Coverage declaration** on every result: which clause types were checked, which skipped, and how much of the document was read.
- **✉️ Negotiation email**: one click turns the flags into a polite, plain-English push-back email (draft for review).
- **Custom UI without a build step**: `gradio.Server` serves vanilla `index.html`, `static/app.css`, and `static/app.js`; if `Server` is unavailable, the app falls back to a styled Gradio Blocks interface.

## Run it on your own machine (offline)

```bash
ollama pull hf.co/giladam01/lease-lens-legal-3b-gguf   # 3.4 GB, llama.cpp runtime
```

Training: Unsloth QLoRA on [CUAD](https://huggingface.co/datasets/chenghao/cuad_qa) (CC-BY-4.0) — free Colab T4 for v1/v2, **Modal A100** for v2.5.

> **Not legal advice.** Every flag is a draft for review — Lease Lens describes what a clause does, not what you should sign.
