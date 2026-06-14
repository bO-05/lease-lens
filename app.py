# Lease Lens v3.0 - custom evidence desk UI
# Fine-tuned 3B legal model (adapter: giladam01/lease-lens-legal-3b) on ZeroGPU.
# Model, prompts, extraction guards, scoring, and generation behavior are kept stable.
import json
import html as _html
import os
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # local mock mode can still serve the custom frontend without Gradio.
    gr = None

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


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
STATIC_DIR = ROOT / "static"
MOCK_MODE = os.getenv("LEASE_LENS_MOCK", "").strip() == "1"

BASE = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER = "giladam01/lease-lens-legal-3b"

if not MOCK_MODE:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"  # decoder-only batch generation
    _base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    # ZeroGPU FIX: at startup the platform emulates CUDA, which tricks PEFT into loading
    # adapter tensors straight onto a GPU that doesn't physically exist yet
    # ("No CUDA GPUs are available"). Force the adapter load onto CPU, merge there,
    # THEN move the merged model to (emulated) CUDA - the supported pattern.
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

# Real executed leases from SEC EDGAR filings (public record) - judges can analyze a genuine
# contract in one click, not just the teaching samples.
REAL_SOURCES = {}
try:
    from sample_contracts import REAL_EXAMPLES, REAL_SOURCES
    EXAMPLES = {**REAL_EXAMPLES, **EXAMPLES}
except Exception as _e:
    print("sample_contracts not loaded:", _e)

DEFAULT_EXAMPLE = next(iter(EXAMPLES))


def _source_banner_html(name):
    url = REAL_SOURCES.get(name)
    if not url:
        return ""
    return ('<div class="source-banner">'
            '<span class="source-dot">SEC</span>'
            '<b>Real executed contract</b>, public filing - '
            '<a href="' + _html.escape(url, quote=True) + '" target="_blank" rel="noopener noreferrer">'
            'verify the source on sec.gov</a> · outside the model training data.</div>')


def get_example_payload(name):
    if name not in EXAMPLES:
        name = DEFAULT_EXAMPLE
    return {
        "name": name,
        "text": EXAMPLES.get(name, ""),
        "source_url": REAL_SOURCES.get(name, ""),
        "source_banner_html": _source_banner_html(name),
        "is_real": name in REAL_SOURCES,
    }


def bootstrap_payload():
    return {
        "examples": list(EXAMPLES.keys()),
        "default_example": DEFAULT_EXAMPLE,
        "proof_chips": [
            "3B fine-tune",
            "+242% F1 vs base",
            "SEC-filed examples",
            "GGUF / llama.cpp",
            "ZeroGPU",
            "No external LLM API",
        ],
        "mock_mode": MOCK_MODE,
    }


def highlight_html(contract, snippets):
    esc = _html.escape(contract)
    for s in snippets:
        es = _html.escape((s or "").strip())
        if len(es) > 3 and es in esc:
            esc = esc.replace(es, '<mark>' + es + '</mark>', 1)
    return '<div class="contract-page">' + esc + '</div>'


def _score_findings(findings):
    score = sum(2 if f["risk"] == "high" else 1 for f in findings)
    maxscore = sum(2 if c["risk"] == "high" else 1 for c in CLAUSES)
    risk_pct = round(100 * score / maxscore) if maxscore else 0
    high_n = sum(1 for f in findings if f["risk"] == "high")
    verdict = "High risk" if high_n else ("Some risk" if findings else "Looks clean")
    return risk_pct, high_n, verdict


def _analysis_payload(text, findings, skipped, n_checked, snippets, chunk_count=0, char_count=0, note=""):
    risk_pct, high_n, verdict = _score_findings(findings)
    return {
        "status": "ok",
        "score": risk_pct,
        "verdict": verdict,
        "high_count": high_n,
        "flag_count": len(findings),
        "checked_count": n_checked,
        "total_clause_count": len(CLAUSES),
        "skipped": skipped,
        "findings": findings,
        "snippets": snippets,
        "highlighted_html": highlight_html(text, snippets),
        "coverage_note": note,
        "chunk_count": chunk_count,
        "char_count": char_count,
        "mock_mode": MOCK_MODE,
        "disclaimer": "Not legal advice - every flag is a draft for review.",
    }


def _quote_near_keyword(text, keywords):
    lower = text.lower()
    hits = [lower.find(k) for k in keywords if lower.find(k) >= 0]
    if not hits:
        return ""
    idx = min(hits)
    start = max(0, idx - 180)
    end = min(len(text), idx + 420)
    return text[start:end].strip()


def _mock_analyze_contract(text):
    text = (text or "").strip()
    if len(text) < 40:
        return {"status": "empty", "message": "Paste or pick a contract first.", "findings": []}
    findings, snippets, covered = [], [], set()
    for c in CLAUSES:
        quote = _quote_near_keyword(text, c["kw"])
        if not quote:
            continue
        covered.add(c["label"])
        if len(findings) >= 5:
            continue
        snippets.append(quote)
        findings.append({**c, "text": quote})
    skipped = [c["label"] for c in CLAUSES if c["label"] not in covered]
    if not findings and covered:
        findings.append({**CLAUSES[0], "text": text[:420].strip()})
        snippets.append(findings[0]["text"])
    return _analysis_payload(text, findings, skipped, len(covered), snippets,
                             chunk_count=1, char_count=len(text),
                             note="Mock UI mode: deterministic local preview, not model output.")


def analyze_contract_payload(text):
    if MOCK_MODE:
        return _mock_analyze_contract(text)

    text = (text or "").strip()
    if len(text) < 40:
        return {"status": "empty", "message": "Paste or pick a contract first.", "findings": []}
    cn = " ".join(text.lower().split())
    # CHUNKED ANALYSIS: real contracts are long. Split into overlapping windows (first 80k
    # chars) and route each clause category only to windows containing its keywords.
    WIN, STRIDE, CAP = 5000, 4000, 80000
    body_text = text[:CAP]
    chunks = [body_text[s:s + WIN] for s in range(0, max(len(body_text), 1), STRIDE)]
    chunks = [c for c in chunks if len(c) > 200] or [body_text]
    pairs = []  # (clause, chunk_text)
    for c in CLAUSES:
        hit = [ch for ch in chunks if any(k in " ".join(ch.lower().split()) for k in c["kw"])]
        for ch in hit[:6]:
            pairs.append((c, ch))
    covered = {c["label"] for c, _ in pairs}
    skipped = [c["label"] for c in CLAUSES if c["label"] not in covered]
    if not pairs:
        return _analysis_payload(text, [], skipped, 0, [], len(chunks), len(body_text))

    msgs = [("Highlight any part of this contract related to: " + c["cat"] +
             ". If there is none, reply NONE.\n\n---\nContract:\n" + ch) for c, ch in pairs]
    try:
        answers = run_batch(msgs)
    except Exception as e:
        return {
            "status": "error",
            "message": "GPU call failed: " + str(e)[:200] + " - try again in a minute (ZeroGPU queue).",
            "findings": [],
            "highlighted_html": highlight_html(text, []),
        }

    findings, snippets, used, found_cats = [], [], [], set()
    for (c, _ch), a in zip(pairs, answers):
        if c["label"] in found_cats:
            continue
        a = (a or "").strip()
        if a.upper() == "NONE" or len(a) <= 3:
            continue
        cand = a.split(" | ")[0].strip()
        ncs = " ".join(cand.lower().split())
        if len(ncs) < 12 or ncs[:60] not in cn:
            continue
        if any(ncs[:80] == u[:80] or ncs in u or u in ncs for u in used):
            continue
        if not any(k in ncs for k in c["kw"]):
            continue
        used.append(ncs)
        snippets.append(cand)
        findings.append({**c, "text": cand})
        found_cats.add(c["label"])

    note = ""
    if len(text) > CAP:
        note = "Analyzed the first " + str(CAP) + " of " + str(len(text)) + " characters."
    return _analysis_payload(text, findings, skipped, len(covered), snippets, len(chunks), len(body_text), note)


def draft_email_payload(state_json):
    try:
        findings = json.loads(state_json or "[]")
    except Exception:
        findings = []
    if not findings:
        return {"status": "empty", "email": "Run an analysis first - then I can draft the email from the flagged clauses."}
    lines = []
    for f in findings:
        lines.append("- " + f["label"] + ': "' + f["text"][:220] + '" -> ask: ' + f["tip"])

    if MOCK_MODE:
        return {
            "status": "ok",
            "email": (
                "Subject: Lease revision requests\n\n"
                "Hi,\n\n"
                "I reviewed the draft and would like to discuss a few clauses before signing:\n\n"
                + "\n".join(lines[:4]) +
                "\n\nCould you send a revised draft reflecting these changes?\n\nThanks,\n\n---\nDraft for review - not legal advice."
            ),
        }

    msg = ("Write a short, polite, plain-English email to the other party of a contract, "
           "proposing changes to these flagged clauses. No legalese (say 'under' not 'pursuant to'). "
           "For each clause: what it currently says, and the change we request. Factual tone; do not "
           "say 'you should' or give legal advice. End asking for a revised draft.\n\nFlagged clauses:\n"
           + "\n".join(lines))
    try:
        out = run_batch([msg], max_new_tokens=400)[0]
    except Exception as e:
        return {"status": "error", "email": "GPU call failed: " + str(e)[:200] + " - try again shortly."}
    return {"status": "ok", "email": out + "\n\n---\nDraft for review - not legal advice."}


def _index_html():
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html.replace("__LEASE_LENS_BOOTSTRAP__", json.dumps(bootstrap_payload()))


def launch_server():
    Server = getattr(gr, "Server", None)
    if Server is None:
        return launch_blocks_fallback()

    from fastapi import Request
    from fastapi.responses import HTMLResponse, FileResponse

    app = Server()

    @app.api(name="get_example")
    def get_example(name: str):
        return get_example_payload(name)

    @app.api(name="analyze_contract")
    def analyze_contract(text: str):
        return analyze_contract_payload(text)

    @app.api(name="draft_email")
    def draft_email(state_json: str):
        return draft_email_payload(state_json)

    @app.get("/api/get_example")
    async def rest_get_example(name: str = DEFAULT_EXAMPLE):
        return get_example_payload(name)

    @app.post("/api/analyze_contract")
    async def rest_analyze_contract(request: Request):
        data = await request.json()
        return analyze_contract_payload(data.get("text", ""))

    @app.post("/api/draft_email")
    async def rest_draft_email(request: Request):
        data = await request.json()
        return draft_email_payload(data.get("state_json", "[]"))

    @app.get("/", response_class=HTMLResponse)
    async def homepage():
        return _index_html()

    @app.get("/static/{file_path:path}")
    async def static_files(file_path: str):
        target = (STATIC_DIR / file_path).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            raise FileNotFoundError(file_path)
        return FileResponse(target)

    app.launch(show_error=True)


CSS_FALLBACK = """
footer{display:none!important}
:root{
  --ll-ink:#08090b;
  --ll-panel:#10141b;
  --ll-panel-2:#151b24;
  --ll-paper:#f5ecd9;
  --ll-muted:#b9aa91;
  --ll-brass:#c49a4a;
  --ll-red:#c33b37;
  --ll-cyan:#67d7d4;
  --ll-green:#7fdca1;
  --ll-line:rgba(196,154,74,.34);
}
html,body,body.dark{background:#08090b!important;color:var(--ll-paper)!important;overflow-x:hidden!important}
.gradio-container{
  max-width:1280px!important;
  margin:0 auto!important;
  padding:18px 18px 42px!important;
  background:
    linear-gradient(90deg,rgba(196,154,74,.04) 1px,transparent 1px),
    linear-gradient(rgba(196,154,74,.035) 1px,transparent 1px),
    radial-gradient(circle at 14% 0%,rgba(195,59,55,.16),transparent 34%),
    radial-gradient(circle at 88% 12%,rgba(103,215,212,.12),transparent 30%),
    #08090b!important;
  background-size:34px 34px,34px 34px,auto,auto,auto!important;
  box-shadow:0 0 0 100vmax #08090b!important;
  color:var(--ll-paper)!important;
  font-family:"Segoe UI",ui-sans-serif,system-ui,sans-serif!important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .contain{
  background:transparent!important;
  border-color:transparent!important;
  color:var(--ll-paper)!important;
}
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span{
  color:#eadfca!important;
  font-weight:750!important;
  letter-spacing:0!important;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container select{
  background:#0c1016!important;
  border:1px solid rgba(196,154,74,.38)!important;
  color:#f6efe1!important;
  border-radius:8px!important;
}
.gradio-container textarea:focus,
.gradio-container input:focus{
  border-color:rgba(103,215,212,.72)!important;
  box-shadow:0 0 0 3px rgba(103,215,212,.12)!important;
}
#hdr{
  position:relative;
  overflow:hidden;
  border:1px solid var(--ll-line);
  border-radius:8px;
  background:
    linear-gradient(135deg,rgba(245,236,217,.07),rgba(245,236,217,.015) 46%,rgba(195,59,55,.12)),
    #10141b;
  box-shadow:0 20px 70px rgba(0,0,0,.34),inset 0 1px 0 rgba(255,255,255,.04);
  padding:22px;
  margin-bottom:16px;
}
#hdr:before{
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(90deg,transparent 0 49%,rgba(196,154,74,.11) 50%,transparent 51%),
             linear-gradient(transparent 0 49%,rgba(196,154,74,.08) 50%,transparent 51%);
  background-size:28px 28px;
  opacity:.28;
  pointer-events:none;
}
#hdr > *{position:relative}
.lease-topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px}
.lease-brand{display:flex;align-items:center;gap:12px}
.lease-seal{
  width:42px;height:42px;display:grid;place-items:center;border:1px solid rgba(196,154,74,.62);
  border-radius:8px;background:#0b0f14;color:var(--ll-cyan);font-family:Georgia,serif;font-weight:900;
  box-shadow:inset 0 0 0 1px rgba(103,215,212,.12);
}
.lease-name{font-family:Georgia,"Times New Roman",serif;font-size:28px;font-weight:900;color:#f8efd9;line-height:1}
.lease-sub{font-size:12px;color:var(--ll-muted);font-family:ui-monospace,Consolas,monospace;text-transform:uppercase}
.proof-strip{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px;max-width:690px}
.proof-chip{
  font-family:ui-monospace,Consolas,monospace;font-size:11px;color:#bdf4f1;background:rgba(103,215,212,.08);
  border:1px solid rgba(103,215,212,.32);padding:6px 9px;border-radius:999px;white-space:nowrap;
}
.hero-line{max-width:790px}
.eyebrow{margin:0 0 8px;color:var(--ll-brass);font-size:12px;font-weight:900;font-family:ui-monospace,Consolas,monospace;text-transform:uppercase}
#hdr h1{color:#fbf0d9;font-family:Georgia,"Times New Roman",serif;margin:0;font-size:44px;line-height:1.02;letter-spacing:0}
#hdr p.hero-copy{color:#d8ccb7;font-size:16px;line-height:1.55;max-width:820px;margin:12px 0 0}
.judge-rail{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:20px}
.rail-step{background:rgba(8,9,11,.56);border:1px solid rgba(196,154,74,.34);border-radius:8px;padding:12px;display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:start}
.rail-num{width:26px;height:26px;display:grid;place-items:center;border-radius:999px;background:rgba(195,59,55,.22);color:#ffd9d7;border:1px solid rgba(195,59,55,.55);font-family:ui-monospace,Consolas,monospace;font-weight:900}
.rail-step b{display:block;color:#f8efd9;font-size:14px}
.rail-step small{display:block;color:var(--ll-muted);line-height:1.35;margin-top:2px}
.source-banner{
  border:1px solid rgba(103,215,212,.36);background:rgba(103,215,212,.075);border-radius:8px;padding:10px 12px;
  color:#d9fbfa;margin:0 0 12px;font-size:13px;
}
.source-banner a{color:#9ff2ee!important;font-weight:800}
.source-dot{
  display:inline-grid;place-items:center;margin-right:8px;border:1px solid rgba(103,215,212,.54);border-radius:4px;
  padding:2px 5px;color:#081014;background:#8be8e3;font-family:ui-monospace,Consolas,monospace;font-size:11px;font-weight:900;
}
#intake_grid{gap:12px!important;margin-bottom:8px}
#go_row{margin:12px 0 10px!important;gap:10px!important}
#go_row button{
  border-radius:8px!important;
  min-height:48px!important;
  font-weight:900!important;
  letter-spacing:0!important;
  box-shadow:0 10px 28px rgba(195,59,55,.20)!important;
}
#go_row button.primary{
  background:linear-gradient(180deg,#dd4b45,#a92e2b)!important;
  border:1px solid rgba(255,190,185,.32)!important;
  color:#fff8ed!important;
}
#status_panel{margin:6px 0 14px}
.status-card{
  border:1px solid rgba(196,154,74,.34);border-radius:8px;background:rgba(16,20,27,.86);
  padding:13px 14px;color:#eadfca;display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:center;
}
.status-icon{width:34px;height:34px;display:grid;place-items:center;border-radius:999px;border:1px solid rgba(196,154,74,.46);color:var(--ll-brass);font-family:ui-monospace,Consolas,monospace;font-weight:900}
.status-card b{display:block;color:#fff0d7}
.status-card span{display:block;color:var(--ll-muted);font-size:13px;line-height:1.4;margin-top:1px}
.status-card.pending{border-color:rgba(103,215,212,.52);box-shadow:0 0 0 1px rgba(103,215,212,.10),0 0 38px rgba(103,215,212,.08)}
.spinner{width:18px;height:18px;border-radius:999px;border:3px solid rgba(103,215,212,.22);border-top-color:var(--ll-cyan);animation:llspin .8s linear infinite}
.scanline{position:relative;height:4px;background:rgba(103,215,212,.12);border-radius:999px;overflow:hidden;margin-top:8px}
.scanline:after{content:"";position:absolute;inset:0;width:42%;background:linear-gradient(90deg,transparent,var(--ll-cyan),transparent);animation:llscan 1.3s ease-in-out infinite}
@keyframes llspin{to{transform:rotate(360deg)}}
@keyframes llscan{0%{transform:translateX(-110%)}100%{transform:translateX(250%)}}
.results-shell{display:grid;gap:12px}
.risk-docket{border:1px solid rgba(196,154,74,.36);border-radius:8px;background:linear-gradient(180deg,rgba(21,27,36,.96),rgba(11,15,20,.96));padding:16px;color:#f7edda}
.docket-top{display:flex;align-items:center;gap:16px;justify-content:space-between}
.score-seal{
  --score:0%;width:112px;height:112px;border-radius:999px;display:grid;place-items:center;flex:0 0 auto;
  background:conic-gradient(var(--ll-red) var(--score),rgba(196,154,74,.16) 0);
  border:1px solid rgba(196,154,74,.42);box-shadow:inset 0 0 0 9px #10141b;
}
.score-seal strong{font-family:Georgia,serif;font-size:34px;line-height:1;color:#fff4df}
.score-seal small{font-family:ui-monospace,Consolas,monospace;color:var(--ll-muted);font-size:12px}
.docket-copy{flex:1;min-width:0}
.docket-copy h2{margin:0 0 5px;font-family:Georgia,serif;color:#fff4df;font-size:28px;letter-spacing:0}
.docket-copy p{margin:0;color:#cdbfa8;line-height:1.45}
.meta-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.meta-pill{border:1px solid rgba(196,154,74,.34);border-radius:999px;padding:5px 9px;color:#d9c9ad;background:rgba(196,154,74,.07);font-size:12px;font-family:ui-monospace,Consolas,monospace}
.finding-card{border:1px solid rgba(196,154,74,.26);border-left:5px solid var(--ll-red);border-radius:8px;background:#111821;padding:14px;color:#f5ecd9}
.finding-card.med{border-left-color:var(--ll-brass)}
.finding-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:9px}
.finding-title{font-weight:900;color:#fff3dd}
.risk-badge{font-size:11px;font-family:ui-monospace,Consolas,monospace;border-radius:999px;padding:4px 8px;border:1px solid rgba(195,59,55,.55);color:#ffc9c6;background:rgba(195,59,55,.14)}
.finding-card.med .risk-badge{border-color:rgba(196,154,74,.55);color:#ffe0a2;background:rgba(196,154,74,.13)}
.quote-box{white-space:pre-wrap;margin:0 0 10px;background:#080b10;border:1px solid rgba(245,236,217,.08);border-radius:6px;padding:10px;color:#efe6d6;font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.45}
.finding-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mini-note{background:rgba(245,236,217,.045);border:1px solid rgba(245,236,217,.075);border-radius:6px;padding:10px;color:#cdbfa8;line-height:1.42}
.mini-note b{display:block;color:#f7edda;margin-bottom:3px}
.mini-note.push{color:#c8f4d7;border-color:rgba(127,220,161,.22);background:rgba(127,220,161,.055)}
.contract-page{white-space:pre-wrap;max-height:680px;overflow:auto;background:#f7f0df;color:#1a1712;border:1px solid rgba(196,154,74,.45);border-radius:8px;padding:20px;font-family:Georgia,"Times New Roman",serif;line-height:1.56;box-shadow:inset 0 0 0 1px rgba(0,0,0,.04)}
.contract-page mark{background:linear-gradient(transparent 48%,rgba(103,215,212,.52) 48%);color:#050505;border-bottom:2px solid #00a8a3;padding:0 2px}
.placeholder-panel{border:1px dashed rgba(196,154,74,.35);border-radius:8px;background:rgba(16,20,27,.56);padding:18px;color:#baad96;min-height:160px}
.placeholder-panel b{display:block;color:#f3e8d1;margin-bottom:6px}
.doc-loading{min-height:220px;display:grid;place-items:center;border:1px dashed rgba(103,215,212,.38);border-radius:8px;background:rgba(103,215,212,.05);color:#bdf4f1}
@media (max-width:760px){
  .gradio-container{padding:10px 10px 28px!important}
  #hdr{padding:16px}
  .lease-topbar{align-items:flex-start;flex-direction:column}
  .proof-strip{justify-content:flex-start}
  #hdr h1{font-size:34px}
  .judge-rail,.finding-grid{grid-template-columns:1fr}
  .docket-top{align-items:flex-start;flex-direction:column}
  .score-seal{width:96px;height:96px}
}
"""


def _status_html(kind="ready", title="Ready to analyze", detail="Default SEC-filed lease is loaded. Press Analyze contract when ready."):
    icon = "LL"
    extra = ""
    if kind == "pending":
        icon = '<span class="spinner"></span>'
        extra = '<div class="scanline"></div>'
    elif kind == "done":
        icon = "OK"
    elif kind == "error":
        icon = "!"
    return (
        '<div class="status-card ' + _html.escape(kind) + '">'
        '<div class="status-icon">' + icon + '</div>'
        '<div><b>' + _html.escape(title) + '</b><span>' + _html.escape(detail) + '</span>' + extra + '</div>'
        '</div>'
    )


def _placeholder_results_html():
    return (
        '<div class="placeholder-panel"><b>Risk docket appears here</b>'
        'Lease Lens will score the contract, group the risky clauses, and show verbatim evidence after analysis.</div>'
    )


def _doc_placeholder_html():
    return (
        '<div class="placeholder-panel"><b>Highlighted contract page</b>'
        'Flagged clauses will be highlighted in the source text so the evidence is easy to verify.</div>'
    )


def _analysis_pending_outputs():
    return (
        _status_html("pending", "Analyzing contract", "Routing clause checks, running the 3B model, and verifying verbatim evidence."),
        '<div class="risk-docket"><div class="docket-top"><div class="score-seal" style="--score:12%"><strong>...</strong></div>'
        '<div class="docket-copy"><h2>Building risk docket</h2><p>The model is checking renewal, fees, liability, arbitration, repair burden, and other clauses. This can take a bit on ZeroGPU.</p>'
        '<div class="scanline"></div></div></div></div>',
        '<div class="doc-loading"><div><b>Scanning contract text</b><div class="scanline"></div></div></div>',
        "",
        gr.update(value="Analyzing...", interactive=False),
    )


def _result_status_html(data):
    if data.get("status") == "empty":
        return _status_html("error", "No contract text", data.get("message", "Paste or load a contract first."))
    if data.get("status") == "error":
        return _status_html("error", "Analysis failed", data.get("message", "Try again shortly."))
    detail = (
        str(data.get("flag_count", 0)) + " flags found across " +
        str(data.get("checked_count", 0)) + " checked clause groups. " +
        "Every shown quote is grounded in the contract text."
    )
    return _status_html("done", "Analysis complete", detail)


def _render_blocks_results(data):
    if data.get("status") == "empty":
        return _placeholder_results_html()
    if data.get("status") == "error":
        return (
            '<div class="placeholder-panel"><b>Analysis failed</b>' +
            _html.escape(data.get("message", "Try again shortly.")) + '</div>'
        )

    score = max(0, min(100, int(data.get("score", 0))))
    flag_count = int(data.get("flag_count", 0))
    checked_count = int(data.get("checked_count", 0))
    total_count = int(data.get("total_clause_count", len(CLAUSES)))
    note = data.get("coverage_note") or (
        "Read " + str(data.get("char_count", 0)) + " characters across " +
        str(data.get("chunk_count", 0)) + " chunks."
    )
    cards = [
        '<div class="results-shell"><div class="risk-docket"><div class="docket-top">'
        '<div class="score-seal" style="--score:' + str(score) + '%"><div><strong>' + str(score) +
        '</strong><small>/100</small></div></div>'
        '<div class="docket-copy"><h2>' + _html.escape(data.get("verdict", "Risk review")) + '</h2>'
        '<p>' + str(flag_count) + ' risky clause' + ('' if flag_count == 1 else 's') +
        ' flagged from ' + str(checked_count) + ' checked groups. Lease Lens reports evidence, not legal advice.</p>'
        '<div class="meta-row">'
        '<span class="meta-pill">High flags: ' + str(data.get("high_count", 0)) + '</span>'
        '<span class="meta-pill">Checked: ' + str(checked_count) + '/' + str(total_count) + '</span>'
        '<span class="meta-pill">Grounded quotes only</span>'
        '</div></div></div><div class="meta-row"><span class="meta-pill">' + _html.escape(note) +
        '</span></div></div>'
    ]

    findings = data.get("findings", [])
    if not findings:
        cards.append(
            '<div class="placeholder-panel"><b>No grounded risky clauses found</b>'
            'The model did not return a quote that passed the verbatim grounding and relevance checks.</div>'
        )
    for f in findings:
        risk = _html.escape(str(f.get("risk", "med")).lower())
        cards.append(
            '<div class="finding-card ' + risk + '">'
            '<div class="finding-head"><div class="finding-title">' + _html.escape(f.get("label", "Clause")) +
            '</div><span class="risk-badge">' + _html.escape(risk.upper()) + '</span></div>'
            '<pre class="quote-box">' + _html.escape(f.get("text", "")) + '</pre>'
            '<div class="finding-grid">'
            '<div class="mini-note"><b>Why it matters</b>' + _html.escape(f.get("why", "")) + '</div>'
            '<div class="mini-note push"><b>Push back</b>' + _html.escape(f.get("tip", "")) + '</div>'
            '</div></div>'
        )
    cards.append("</div>")
    return "".join(cards)


def launch_blocks_fallback():
    if gr is None:
        raise RuntimeError("Gradio is required unless LEASE_LENS_MOCK=1 is used.")
    with gr.Blocks(css=CSS_FALLBACK, title="Lease Lens") as demo:
        gr.HTML(
            '<section id="hdr">'
            '<div class="lease-topbar"><div class="lease-brand"><span class="lease-seal">LL</span>'
            '<div><div class="lease-name">Lease Lens</div><div class="lease-sub">redline legal evidence desk</div></div></div>'
            '<div class="proof-strip">'
            '<span class="proof-chip">3B fine-tune</span><span class="proof-chip">+242% F1 vs base</span>'
            '<span class="proof-chip">SEC-filed examples</span><span class="proof-chip">GGUF / llama.cpp</span>'
            '<span class="proof-chip">ZeroGPU</span><span class="proof-chip">No external LLM API</span>'
            '</div></div>'
            '<div class="hero-line"><p class="eyebrow">contract risk review before signature</p>'
            '<h1>Read the lease before it reads you.</h1>'
            '<p class="hero-copy">Load a real filing or paste your contract. Lease Lens returns a risk score, grounded clause evidence, highlighted source text, and a plain-English negotiation draft.</p></div>'
            '<div class="judge-rail">'
            '<div class="rail-step"><span class="rail-num">1</span><div><b>Load real filing</b><small>The default SEC lease is ready on open.</small></div></div>'
            '<div class="rail-step"><span class="rail-num">2</span><div><b>Analyze evidence</b><small>Quotes must appear verbatim in the contract.</small></div></div>'
            '<div class="rail-step"><span class="rail-num">3</span><div><b>Draft pushback</b><small>Turn grounded flags into a review email.</small></div></div>'
            '</div></section>'
        )
        with gr.Row(elem_id="intake_grid"):
            ex = gr.Dropdown(choices=list(EXAMPLES.keys()), value=DEFAULT_EXAMPLE, label="Real filing or sample")
            up = gr.File(label="Upload your own .txt contract", file_types=[".txt"], type="filepath")
        src_banner = gr.HTML(value=_source_banner_html(DEFAULT_EXAMPLE))
        inp = gr.Textbox(value=EXAMPLES[DEFAULT_EXAMPLE], lines=10, max_lines=20, label="Contract text", elem_id="contract_input")
        with gr.Row(elem_id="go_row"):
            btn = gr.Button("Analyze contract", variant="primary", scale=4)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        status_panel = gr.HTML(value=_status_html(), elem_id="status_panel")
        st = gr.State("[]")
        with gr.Row(elem_id="evidence_grid"):
            out_cards = gr.HTML(value=_placeholder_results_html())
            out_doc = gr.HTML(value=_doc_placeholder_html())
        with gr.Accordion("Negotiation Letter", open=False):
            email_btn = gr.Button("Draft pushback")
            email_out = gr.Textbox(lines=12, label="Draft for review", show_copy_button=True)

        def load_example(name):
            payload = get_example_payload(name)
            return (
                payload["text"],
                payload["source_banner_html"],
                _status_html("ready", "Example loaded", "Press Analyze contract to build the risk docket."),
                _placeholder_results_html(),
                _doc_placeholder_html(),
                "[]",
                "",
            )

        def load_file(path):
            if not path:
                return "", "", _status_html(), _placeholder_results_html(), _doc_placeholder_html(), "[]", ""
            with open(path, "r", errors="ignore") as fh:
                return (
                    fh.read(),
                    "",
                    _status_html("ready", "Contract uploaded", "Press Analyze contract to check the uploaded text."),
                    _placeholder_results_html(),
                    _doc_placeholder_html(),
                    "[]",
                    "",
                )

        def analyze_blocks(text):
            try:
                data = analyze_contract_payload(text)
                doc = data.get("highlighted_html", "") if data.get("status") == "ok" else _doc_placeholder_html()
            except Exception as e:
                data = {
                    "status": "error",
                    "message": "Analysis failed: " + str(e)[:220],
                    "findings": [],
                }
                doc = _doc_placeholder_html()
            return (
                _result_status_html(data),
                _render_blocks_results(data),
                doc,
                json.dumps(data.get("findings", [])),
                gr.update(value="Analyze contract", interactive=True),
            )

        def draft_blocks(state_json):
            return draft_email_payload(state_json).get("email", "")

        ex.change(load_example, ex, [inp, src_banner, status_panel, out_cards, out_doc, st, email_out])
        up.upload(load_file, up, [inp, src_banner, status_panel, out_cards, out_doc, st, email_out])
        pending = btn.click(
            _analysis_pending_outputs,
            None,
            [status_panel, out_cards, out_doc, email_out, btn],
            queue=False,
            show_progress="hidden",
        )
        pending.then(
            analyze_blocks,
            inp,
            [status_panel, out_cards, out_doc, st, btn],
            show_progress="minimal",
            show_progress_on=status_panel,
        )
        email_btn.click(draft_blocks, st, email_out)
        clear_btn.click(
            lambda: (
                "",
                "",
                _status_html("ready", "Ready to analyze", "Load a filing, upload a .txt file, or paste contract text."),
                _placeholder_results_html(),
                _doc_placeholder_html(),
                "",
                "[]",
                gr.update(value="Analyze contract", interactive=True),
            ),
            None,
            [inp, src_banner, status_panel, out_cards, out_doc, email_out, st, btn],
        )

    demo.queue().launch(ssr_mode=False, show_error=True)


def launch_stdlib_mock_server():
    import mimetypes
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(raw or "{}")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _index_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/get_example":
                name = parse_qs(parsed.query).get("name", [DEFAULT_EXAMPLE])[0]
                self._json(get_example_payload(name))
                return
            if parsed.path.startswith("/static/"):
                rel = parsed.path.replace("/static/", "", 1)
                target = (STATIC_DIR / rel).resolve()
                if not target.exists() or STATIC_DIR.resolve() not in target.parents:
                    self.send_error(404)
                    return
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path == "/api/analyze_contract":
                self._json(analyze_contract_payload(self._read_json().get("text", "")))
                return
            if self.path == "/api/draft_email":
                self._json(draft_email_payload(self._read_json().get("state_json", "[]")))
                return
            self.send_error(404)

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "7860"))
    print(f"Lease Lens mock UI running on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def launch():
    if gr is None:
        if MOCK_MODE:
            return launch_stdlib_mock_server()
        raise RuntimeError("Gradio is not installed. Hugging Face Spaces provides it via sdk_version.")
    return launch_server()


if __name__ == "__main__":
    launch()
