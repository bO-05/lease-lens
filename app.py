# Lease Lens v2.0 — ZeroGPU edition
# Fine-tuned 3B legal model (adapter: giladam01/lease-lens-legal-3b) on @spaces.GPU.
# All clause categories run in ONE batched generate -> seconds, not minutes.
import json
import html as _html

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    import spaces  # provided on HF Spaces (ZeroGPU)
except ImportError:  # local fallback so the file also runs off-Spaces
    class _S:
        def GPU(self, *a, **k):
            if a and callable(a[0]):
                return a[0]
            def deco(f):
                return f
            return deco
    spaces = _S()

BASE = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER = "giladam01/lease-lens-legal-3b"

tok = AutoTokenizer.from_pretrained(ADAPTER)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "left"  # decoder-only batch generation
_base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
# ZeroGPU FIX: at startup the platform emulates CUDA, which tricks PEFT into loading
# adapter tensors straight onto a GPU that doesn't physically exist yet
# ("No CUDA GPUs are available"). Force the adapter load onto CPU, merge there,
# THEN move the merged model to (emulated) CUDA — the supported pattern.
try:
    model = PeftModel.from_pretrained(_base, ADAPTER, torch_device="cpu")
except TypeError:  # older peft without torch_device kwarg: hide CUDA during the load
    _avail, _cnt = torch.cuda.is_available, torch.cuda.device_count
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    try:
        model = PeftModel.from_pretrained(_base, ADAPTER)
    finally:
        torch.cuda.is_available, torch.cuda.device_count = _avail, _cnt
model = model.merge_and_unload()
model.to("cuda")  # ZeroGPU emulates CUDA at startup; real GPU attaches inside @spaces.GPU
model.eval()

SYSTEM = ("You are a meticulous legal contract analyst. Given a contract excerpt and a clause "
          "category, extract the exact verbatim text of any clause that matches that category. "
          "If no matching clause is present, reply with exactly: NONE. Do not explain. "
          "Return only the clause text or NONE.")

CLAUSES = [
  {"label":"Automatic Renewal","risk":"high","cat":"Renewal Term / automatic renewal","kw":["renew","automatically","successive"],"why":"It can silently roll over into a new term unless you cancel inside a narrow window.","tip":"Ask for a shorter notice window (30 days) or strike auto-renewal."},
  {"label":"Early Termination Penalty","risk":"high","cat":"early termination penalties or forfeited deposit","kw":["terminat","early termination","forfeit","break the lease"],"why":"The real cost of leaving early -- sometimes months of fees or the whole deposit.","tip":"Negotiate a cap (one month) or a transfer/sublet option."},
  {"label":"Rent / Price Increase","risk":"high","cat":"rent increase or escalation during the term","kw":["increase the rent","rent increase","escalat","market rent","raise the rent","rent may increase"],"why":"Shows whether and how much your cost can climb while locked in.","tip":"Cap increases (e.g. 3%) or tie them to CPI."},
  {"label":"Late Fees & Penalties","risk":"med","cat":"late fees, interest, or penalties for missed payments","kw":["late fee","late payment","overdue","arrears","per day"],"why":"Compounding charges that stack up fast on one missed due date.","tip":"Request a grace period and a flat, reasonable late fee."},
  {"label":"Deposit / Prepayment Terms","risk":"med","cat":"security deposit amount, withholding conditions, and return timeline","kw":["deposit","prepay","advance payment"],"why":"Vague return conditions are the top reason deposits never come back.","tip":"Require an itemized deduction list and a firm return deadline."},
  {"label":"Non-Compete / Exclusivity","risk":"high","cat":"non-compete, exclusivity, or restriction on working with others","kw":["compet","exclusiv","solicit"],"why":"Can limit your ability to earn elsewhere for a long time.","tip":"Narrow the scope, geography, and shorten the duration."},
  {"label":"IP Assignment","risk":"med","cat":"intellectual property ownership or assignment of work product","kw":["intellectual property","copyright","work product","patent","trademark"],"why":"Decides who owns what you create -- sometimes even your prior work.","tip":"Assign only deliverables; carve out your pre-existing IP."},
  {"label":"Liability & Indemnification","risk":"high","cat":"liability shift, indemnification, or hold-harmless / release of liability","kw":["liab","indemnif","harmless","negligence","at own risk"],"why":"Could make you pay for damages or legal costs that are not your fault.","tip":"Make it mutual and carve out the other party's negligence."},
  {"label":"Maintenance / Repairs Burden","risk":"med","cat":"tenant responsibility for repairs, maintenance, or upkeep","kw":["repair","maintenance","maintain","upkeep"],"why":"You may be on the hook for costly repairs that are normally the owner's job.","tip":"Limit your responsibility to damage you cause."},
  {"label":"Arbitration / Jury Waiver","risk":"high","cat":"arbitration requirement, jury-trial waiver, class-action waiver, or governing law","kw":["arbitrat","jury","class action","governing law","waive"],"why":"May quietly strip your right to sue or join a class action.","tip":"Strike the class-action waiver; keep your right to court."},
]


def _chat(user_msg):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@spaces.GPU(duration=120)
def run_batch(user_msgs, max_new_tokens=128):
    """Batched greedy generate (mini-batched internally) -> list of completions."""
    outs = []
    for i in range(0, len(user_msgs), 32):
        prompts = [_chat(m) for m in user_msgs[i:i + 32]]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.decode(g, skip_special_tokens=True).strip() for g in gen)
    return outs


def highlight(contract, snippets):
    esc = _html.escape(contract)
    for s in snippets:
        es = _html.escape(s.strip())
        if len(es) > 3 and es in esc:
            esc = esc.replace(es, '<mark style="background:#ffd86b;color:#1a1a1a;border-radius:3px">' + es + '</mark>', 1)
    return ('<div style="white-space:pre-wrap;font-family:Georgia,\'Iowan Old Style\',serif;font-size:14px;'
            'line-height:1.7;color:#e2e8f0;background:#0d1220;border:1px solid #27314e;padding:16px 18px;'
            'border-radius:10px;max-height:440px;overflow:auto">' + esc + '</div>')


def render_results(findings, skipped, n_checked):
    score = sum(2 if f["risk"] == "high" else 1 for f in findings)
    maxscore = sum(2 if c["risk"] == "high" else 1 for c in CLAUSES)
    risk_pct = round(100 * score / maxscore) if maxscore else 0
    high_n = sum(1 for f in findings if f["risk"] == "high")
    verdict = "High risk" if high_n else ("Some risk" if findings else "Looks clean")
    vcolor = "#fc8181" if high_n else ("#f6ad55" if findings else "#68d391")   # softened for dark mode
    head = ('<div style="font-family:Inter,system-ui,sans-serif;display:flex;gap:20px;align-items:center;'
            'background:#151d30;border:1px solid #27314e;border-radius:14px;padding:16px 20px;margin-bottom:14px">'
            '<div style="font-size:40px;font-weight:700;color:' + vcolor + '">' + str(risk_pct) +
            '<span style="font-size:18px;color:#9aa6c4">/100</span></div>'
            '<div><div style="font-size:18px;font-weight:600;color:' + vcolor + '">' + verdict + '</div>'
            '<div style="color:#9aa6c4;font-size:14px">' + str(len(findings)) + ' clauses flagged (' + str(high_n) +
            ' high-risk) of ' + str(n_checked) + ' checked</div></div></div>')
    cards = []
    for f in sorted(findings, key=lambda x: 0 if x["risk"] == "high" else 1):
        color = "#fc8181" if f["risk"] == "high" else "#f6ad55"
        tag = "High risk" if f["risk"] == "high" else "Review"
        cards.append(
          '<div style="border-left:5px solid ' + color + ';background:rgba(255,255,255,0.05);border-radius:10px;'
          'padding:14px 16px;margin-bottom:14px;color:#e2e8f0;font-family:Inter,system-ui,sans-serif">'
          '<div style="display:flex;gap:10px;align-items:center"><b style="font-size:16px;color:#f8fafc">' + _html.escape(f["label"]) + '</b>'
          '<span style="font-size:11px;color:' + color + ';border:1px solid ' + color + ';border-radius:6px;padding:2px 8px">' + tag + '</span></div>'
          '<div style="font-family:ui-monospace,SFMono-Regular,monospace;font-size:13px;color:#e2e8f0;background:#0d1220;'
          'border-radius:8px;padding:11px 13px;margin:11px 0;white-space:pre-wrap;line-height:1.55">' + _html.escape(f["text"]) + '</div>'
          '<div style="color:#cbd5e0;font-size:14px;line-height:1.5">💡 <b style="color:#90cdf4">Why this matters:</b> ' + _html.escape(f["why"]) + '</div>'
          '<div style="color:#9ae6b4;font-size:14px;margin-top:6px;line-height:1.5">✋ <b style="color:#9ae6b4">Push back:</b> ' + _html.escape(f["tip"]) + '</div></div>')
    body = head + ("".join(cards) if findings else
                   '<p style="color:#5fe0a0;font-family:Inter,sans-serif">No risky clauses flagged.</p>')
    if skipped:
        body += ('<div style="font-family:Inter,system-ui,sans-serif;color:#9aa6c4;font-size:13px;margin-top:10px">'
                 'Coverage: checked ' + str(n_checked) + ' of ' + str(len(CLAUSES)) +
                 ' clause types; skipped (keywords absent): ' + ", ".join(skipped) + '</div>')
    body += ('<div style="font-family:Inter,system-ui,sans-serif;color:#6b7689;font-size:12px;margin-top:8px">'
             'Not legal advice — every flag is a draft for review.</div>')
    return body


def analyze(text):
    text = (text or "").strip()
    if len(text) < 40:
        yield '<p style="color:#ffb653;font-family:Inter,sans-serif">Paste or pick a contract first.</p>', "", "[]"
        return
    cn = " ".join(text.lower().split())
    # CHUNKED ANALYSIS: real contracts are long. Split into overlapping windows (first 80k
    # chars) and route each clause category only to windows containing its keywords.
    WIN, STRIDE, CAP = 5000, 4000, 80000
    body_text = text[:CAP]
    chunks = [body_text[s:s + WIN] for s in range(0, max(len(body_text), 1), STRIDE)]
    chunks = [c for c in chunks if len(c) > 200] or [body_text]
    pairs = []                                   # (clause, chunk_text)
    for c in CLAUSES:
        hit = [ch for ch in chunks if any(k in " ".join(ch.lower().split()) for k in c["kw"])]
        for ch in hit[:6]:                        # cap windows per category
            pairs.append((c, ch))
    covered = {c["label"] for c, _ in pairs}
    skipped = [c["label"] for c in CLAUSES if c["label"] not in covered]
    if not pairs:
        yield ('<p style="color:#5fe0a0;font-family:Inter,sans-serif">No risky clauses flagged '
               '(no clause keywords present in this text).</p>'), highlight(text, []), "[]"
        return
    yield ('<div style="font-family:Inter,system-ui,sans-serif;color:#9aa6c4;background:#151d30;border:1px solid #27314e;'
           'border-radius:12px;padding:12px 16px">⚡ Running ' + str(len(pairs)) + ' checks across ' +
           str(len(chunks)) + ' document windows (' + str(len(body_text)) +
           ' chars) in batched passes on ZeroGPU…</div>'), highlight(text, []), "[]"

    msgs = [("Highlight any part of this contract related to: " + c["cat"] +
             ". If there is none, reply NONE.\n\n---\nContract:\n" + ch) for c, ch in pairs]
    try:
        answers = run_batch(msgs)
    except Exception as e:
        yield ('<div style="color:#ff5d6c;font-family:Inter,sans-serif">GPU call failed: ' +
               _html.escape(str(e)[:200]) + ' — try again in a minute (ZeroGPU queue).</div>'), highlight(text, []), "[]"
        return

    findings, snippets, used, found_cats = [], [], [], set()
    for (c, _ch), a in zip(pairs, answers):
        if c["label"] in found_cats:                                        # first good hit wins
            continue
        a = (a or "").strip()
        if a.upper() == "NONE" or len(a) <= 3:
            continue
        cand = a.split(" | ")[0].strip()
        ncs = " ".join(cand.lower().split())
        if len(ncs) < 12 or ncs[:60] not in cn:                            # grounding (full doc)
            continue
        if any(ncs[:80] == u[:80] or ncs in u or u in ncs for u in used):  # dedup
            continue
        if not any(k in ncs for k in c["kw"]):                             # keyword guard
            continue
        used.append(ncs); snippets.append(cand)
        findings.append({**c, "text": cand}); found_cats.add(c["label"])

    state = json.dumps([{"label": f["label"], "text": f["text"], "tip": f["tip"]} for f in findings])
    extra = ('' if len(text) <= CAP else
             '<div style="font-family:Inter,system-ui,sans-serif;color:#9aa6c4;font-size:13px;margin-top:6px">'
             'Note: analyzed the first ' + str(CAP) + ' of ' + str(len(text)) + ' characters.</div>')
    yield render_results(findings, skipped, len(covered)) + extra, highlight(text, snippets), state


def draft_email(state):
    try:
        findings = json.loads(state or "[]")
    except Exception:
        findings = []
    if not findings:
        return "Run an analysis first — then I can draft the email from the flagged clauses."
    lines = []
    for f in findings:
        lines.append("- " + f["label"] + ': "' + f["text"][:220] + '" -> ask: ' + f["tip"])
    msg = ("Write a short, polite, plain-English email to the other party of a contract, "
           "proposing changes to these flagged clauses. No legalese (say 'under' not 'pursuant to'). "
           "For each clause: what it currently says, and the change we request. Factual tone; do not "
           "say 'you should' or give legal advice. End asking for a revised draft.\n\nFlagged clauses:\n"
           + "\n".join(lines))
    try:
        out = run_batch([msg], max_new_tokens=400)[0]
    except Exception as e:
        return "GPU call failed: " + str(e)[:200] + " — try again shortly."
    return out + "\n\n---\nDraft for review — not legal advice."


EXAMPLES = {
"Apartment lease": """RESIDENTIAL LEASE AGREEMENT (excerpt)

3. TERM. This Lease begins July 1 for twelve (12) months. Unless either party gives written notice at least ninety (90) days before the end of the term, this Lease automatically renews for successive 12-month terms at the then-current market rent.
4. RENT. Monthly rent is $2,400. Landlord may increase the rent by up to eight percent (8%) upon each renewal. Any payment received after the 5th incurs a late fee of $150 plus $25 per day.
7. EARLY TERMINATION. Tenant may terminate early only upon payment of two (2) months' rent, and forfeits the entire security deposit.
11. MAINTENANCE. Tenant is responsible for all repairs and maintenance, including HVAC, plumbing, and appliances, regardless of cause.
15. ENTRY. Landlord may enter the premises at any time to inspect, repair, or show the unit.
19. DISPUTES. The parties waive any right to a jury trial; all disputes shall be resolved by binding arbitration in the county of Landlord's choosing. Tenant waives the right to participate in any class action.""",

"Freelance contract (NDA + IP)": """INDEPENDENT CONTRACTOR AGREEMENT (excerpt)

2. INTELLECTUAL PROPERTY. All work product, and any pre-existing materials incorporated therein, are hereby assigned exclusively to Client upon creation.
3. NON-COMPETE. For twenty-four (24) months after termination, Contractor shall not provide similar services to any competitor of Client anywhere in the United States.
4. PAYMENT. Net-60. Client may terminate at any time without cause, in which case Contractor forfeits any unpaid fees for work in progress.
6. INDEMNIFICATION. Contractor shall indemnify and hold Client harmless from any and all claims arising from the work, regardless of fault.
9. DISPUTES. Any dispute shall be resolved by binding arbitration. Contractor waives the right to a jury trial and to participate in any class or collective action.""",

"SaaS Terms of Service": """SOFTWARE SUBSCRIPTION TERMS (excerpt)

4. TERM & RENEWAL. Your subscription automatically renews for successive annual terms unless cancelled at least sixty (60) days before renewal. We may increase fees by up to fifteen percent (15%) on each renewal.
6. CANCELLATION. To cancel you must send written notice; cancellation is effective at the end of the then-current term. No refunds are provided.
8. LIABILITY. Our total liability is limited to the fees you paid in the one (1) month preceding the claim. We may modify or discontinue the Service at any time without notice.
12. DISPUTES. You agree to binding arbitration and waive any right to a jury trial or to participate in a class action.""",

"Gym membership": """FITNESS MEMBERSHIP AGREEMENT (excerpt)

1. TERM. Twelve (12) month minimum commitment. After the initial term, membership automatically continues month-to-month at the then-current rate.
2. CANCELLATION. Cancellation requires thirty (30) days' written notice delivered by certified mail. Early cancellation within the initial term requires payment of all remaining monthly dues.
3. FEES. Dues are billed monthly. A late payment incurs a $40 fee plus any collection costs and reasonable attorneys' fees.
5. RELEASE OF LIABILITY. Member uses all facilities at Member's own risk and releases the Gym from any and all liability, including claims arising from the Gym's own negligence.""",
}

CSS = """
footer{display:none!important}
.gradio-container{background:#0d1220!important; max-width:1180px!important}
#hdr h1{color:#eaeefb; font-family:Inter,system-ui,sans-serif; margin-bottom:0}
#hdr p{color:#9aa6c4; font-family:Inter,system-ui,sans-serif}
#hdr .chip{display:inline-block;font-family:monospace;font-size:12px;color:#39d3c5;background:rgba(57,211,197,.1);
  border:1px solid rgba(57,211,197,.35);padding:4px 10px;border-radius:999px;margin-right:6px}
#go_row{margin:14px 0}
"""

EMPTY_STATE = ('<div style="font-family:Inter,system-ui,sans-serif;color:#9aa6c4;text-align:center;'
               'background:rgba(255,255,255,0.03);border:1px dashed #27314e;border-radius:12px;padding:34px 20px">'
               '<div style="font-size:34px;margin-bottom:8px">🔍</div>'
               'Pick an example (or upload a contract) above and press <b style="color:#e2e8f0">⚡ Analyze contract</b> '
               'to see the risk score, flagged clauses, and highlights here.</div>')


# Real executed leases from SEC EDGAR filings (public record) — judges can analyze a genuine
# contract in one click, not just the teaching samples.
REAL_SOURCES = {}
try:
    from sample_contracts import REAL_EXAMPLES, REAL_SOURCES
    EXAMPLES = {**REAL_EXAMPLES, **EXAMPLES}
except Exception as _e:
    print("sample_contracts not loaded:", _e)


def _source_banner(name):
    url = REAL_SOURCES.get(name)
    if not url:
        return ""   # synthetic teaching sample — no source banner
    return ('<div style="font-family:Inter,system-ui,sans-serif;font-size:13px;color:#90cdf4;'
            'background:rgba(144,205,244,0.08);border:1px solid rgba(144,205,244,0.3);'
            'border-radius:10px;padding:10px 14px;margin-bottom:8px">'
            '📄 <b>Real executed contract</b>, public SEC filing — '
            '<a href="' + url + '" target="_blank" rel="noopener noreferrer" '
            'style="color:#90cdf4;text-decoration:underline">verify the source on sec.gov ↗</a> · '
            'outside the model\'s training data.</div>')


def load_example(name):
    # returns (contract_text, source_banner_html)
    return EXAMPLES.get(name, ""), _source_banner(name)


def load_file(path):
    if not path:
        return "", ""
    with open(path, "r", errors="ignore") as fh:
        return fh.read(), ""   # uploaded file: no SEC source banner


DEFAULT_EXAMPLE = next(iter(EXAMPLES))


# Gradio 5.x (pinned via README sdk_version): css belongs in the Blocks constructor.
with gr.Blocks(css=CSS, title="Lease Lens") as demo:
    gr.HTML('<div id="hdr"><h1>🔍 Lease Lens</h1>'
            '<p>Paste a lease or contract — a <b>fine-tuned 3B legal model</b> scores the risk, flags clauses '
            'verbatim, highlights them in the text, and drafts your negotiation email.</p>'
            '<span class="chip">⚡ ZeroGPU · seconds per analysis</span>'
            '<span class="chip">3B fine-tune · +242% F1 vs base</span>'
            '<span class="chip">also ships as GGUF for llama.cpp</span></div>')
    with gr.Row():
        ex = gr.Dropdown(choices=list(EXAMPLES.keys()), value=DEFAULT_EXAMPLE, label="Load a real-world example")
        up = gr.File(label="…or upload your own .txt contract", file_types=[".txt"], type="filepath")
    src_banner = gr.HTML(value=_source_banner(DEFAULT_EXAMPLE))   # shows SEC provenance when a real lease is selected
    inp = gr.Textbox(value=EXAMPLES[DEFAULT_EXAMPLE], lines=10, max_lines=20, label="Contract text")
    with gr.Row(elem_id="go_row"):
        btn = gr.Button("⚡ Analyze contract", variant="primary", scale=4)
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)
    st = gr.State("[]")
    with gr.Row():
        out_cards = gr.HTML(value=EMPTY_STATE)
        out_doc = gr.HTML()
    with gr.Accordion("✉️ Draft a negotiation email from the flags", open=False):
        email_btn = gr.Button("Draft email")
        email_out = gr.Textbox(lines=12, label="Draft (copy-paste, edit before sending)", show_copy_button=True)
    ex.change(load_example, ex, [inp, src_banner])
    up.upload(load_file, up, [inp, src_banner])
    btn.click(analyze, inp, [out_cards, out_doc, st])
    email_btn.click(draft_email, st, email_out)
    clear_btn.click(lambda: ("", "", EMPTY_STATE, "", "[]"), None, [inp, src_banner, out_cards, out_doc, st])

demo.queue().launch(ssr_mode=False, show_error=True)
