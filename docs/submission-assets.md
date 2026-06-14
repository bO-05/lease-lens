# Submission Assets

Use this as the fast publishing checklist for the Build Small Hackathon final push.

## Demo Video Script

Target length: 60-90 seconds.

1. Open the live Space:
   https://huggingface.co/spaces/build-small-hackathon/lease-lens
2. Say: "Lease Lens is a 3B legal model that reads a lease before you sign it. It runs inside the Hugging Face Space with no external LLM API."
3. Leave the default real SEC-filed lease selected and point out the SEC source banner.
4. Click **Analyze contract**.
5. Show the risk score, number of flags, and at least one verbatim highlighted clause.
6. Open the negotiation email accordion and click **Draft email**.
7. End on proof:
   "It is a Llama-3.2-3B fine-tune, measured at +242% relative F1 over base on held-out CUAD items, ships as GGUF for llama.cpp, and the final submission repo contains Codex-attributed commits."

Suggested title:

```text
Lease Lens - a 3B legal model that reads the lease before you sign it
```

Suggested description:

```text
Build Small Hackathon submission: Lease Lens flags risky clauses in leases and contracts using a fine-tuned 3B model running inside a Gradio Space. It shows verbatim grounded clauses, risk score, highlights, and a negotiation email draft. No external LLM API is called. Model, GGUF, field notes, and Codex-attributed GitHub repo are linked from the Space README.
```

## Social Post Copy

```text
I built Lease Lens for the Hugging Face Build Small Hackathon: a 3B legal model that reads a lease before you sign it.

Paste a contract and it returns:
- verbatim risky-clause flags
- a 0-100 risk score
- in-text highlights
- plain-English pushback tips
- a negotiation email draft

It runs in a Gradio Space with no external LLM API, ships as a GGUF for llama.cpp/Ollama, and was tested on real SEC-filed leases outside the training set.

Main track: Backyard AI
Targets: OpenAI Codex Track, Tiny Titan, Well-Tuned, Off the Grid, Llama Champion, Field Notes, Best Use of Modal, Best Demo, Bonus Quest Champion.

Space: https://huggingface.co/spaces/build-small-hackathon/lease-lens
Model: https://huggingface.co/giladam01/lease-lens-legal-3b
GGUF: https://huggingface.co/giladam01/lease-lens-legal-3b-gguf
Field notes: https://huggingface.co/blog/giladam01/lease-lens-article
GitHub/Codex repo: [PUBLIC_GITHUB_REPO_URL]
Demo: [DEMO_VIDEO_URL]
```

## Final README Replacement Checklist

Replace these placeholders before final submission:

- `[PUBLIC_GITHUB_REPO_URL]`
- `[DEMO_VIDEO_URL]`
- `[SOCIAL_POST_URL]`

After those links are live, add `achievement:sharing` to the README front matter if the Codex build log is public in the GitHub repo or on the Hub.

## GitHub Publish Commands

Run these from the repository root after authenticating GitHub CLI:

```powershell
gh auth login
gh repo create lease-lens --public --source . --remote origin --push
```

Then replace `[PUBLIC_GITHUB_REPO_URL]` in `README.md` with the URL returned by `gh`, commit that README update, and push it:

```powershell
git add README.md
git commit -m "docs: add public codex repo link" -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
git push origin main
```

If Git reports dubious ownership on this Windows workspace, run the same commands with:

```powershell
git -c safe.directory=D:/Repo/lease-lens/lease-lens-space-FINAL <command>
```
