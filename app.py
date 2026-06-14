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
.gradio-container{background:#090b0f!important; max-width:1240px!important}
#hdr{background:#11151d;border:1px solid #52442c;border-radius:8px;padding:20px 22px;margin-bottom:14px}
#hdr h1{color:#f4ead7;font-family:Georgia,serif;margin:0 0 4px;font-size:34px}
#hdr p{color:#c9bdab;font-family:Segoe UI,sans-serif;max-width:760px}
#hdr .chip{display:inline-block;font-family:ui-monospace,monospace;font-size:12px;color:#6bd7d2;background:rgba(107,215,210,.09);
  border:1px solid rgba(107,215,210,.35);padding:5px 10px;border-radius:999px;margin:5px 6px 0 0}
#go_row{margin:14px 0}
"""


def _render_blocks_results(data):
    if data.get("status") == "empty":
        return '<div style="color:#e3b15f;font-family:Segoe UI,sans-serif">' + data["message"] + '</div>'
    if data.get("status") == "error":
        return '<div style="color:#ff6b6b;font-family:Segoe UI,sans-serif">' + _html.escape(data["message"]) + '</div>'
    cards = [
        '<div style="background:#11151d;border:1px solid #52442c;border-radius:8px;padding:16px;color:#f4ead7;font-family:Segoe UI,sans-serif;margin-bottom:12px">'
        '<div style="font-family:Georgia,serif;font-size:38px;color:#b73737">' + str(data["score"]) + '<span style="font-size:16px;color:#9f927d">/100</span></div>'
        '<b>' + _html.escape(data["verdict"]) + '</b><br>'
        '<span style="color:#9f927d">' + str(data["flag_count"]) + ' clauses flagged of ' + str(data["checked_count"]) + ' checked</span></div>'
    ]
    for f in data.get("findings", []):
        cards.append('<div style="border-left:4px solid #b73737;background:#121821;border-radius:8px;padding:13px;margin-bottom:10px;color:#f4ead7">'
                     '<b>' + _html.escape(f["label"]) + '</b>'
                     '<pre style="white-space:pre-wrap;color:#e5dccd;background:#090b0f;padding:10px;border-radius:6px">' + _html.escape(f["text"]) + '</pre>'
                     '<div style="color:#c9bdab">Why: ' + _html.escape(f["why"]) + '</div>'
                     '<div style="color:#84d39b">Push back: ' + _html.escape(f["tip"]) + '</div></div>')
    return "".join(cards)


def launch_blocks_fallback():
    if gr is None:
        raise RuntimeError("Gradio is required unless LEASE_LENS_MOCK=1 is used.")
    with gr.Blocks(css=CSS_FALLBACK, title="Lease Lens") as demo:
        gr.HTML('<div id="hdr"><h1>Lease Lens</h1>'
                '<p>Read the lease before it reads you. A fine-tuned 3B legal model scores risk, flags verbatim clauses, highlights evidence, and drafts pushback.</p>'
                '<span class="chip">3B fine-tune</span><span class="chip">+242% F1 vs base</span>'
                '<span class="chip">SEC-filed examples</span><span class="chip">GGUF / llama.cpp</span></div>')
        with gr.Row():
            ex = gr.Dropdown(choices=list(EXAMPLES.keys()), value=DEFAULT_EXAMPLE, label="Load a real filing or sample")
            up = gr.File(label="...or upload your own .txt contract", file_types=[".txt"], type="filepath")
        src_banner = gr.HTML(value=_source_banner_html(DEFAULT_EXAMPLE))
        inp = gr.Textbox(value=EXAMPLES[DEFAULT_EXAMPLE], lines=10, max_lines=20, label="Contract text")
        with gr.Row(elem_id="go_row"):
            btn = gr.Button("Analyze contract", variant="primary", scale=4)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        st = gr.State("[]")
        with gr.Row():
            out_cards = gr.HTML()
            out_doc = gr.HTML()
        with gr.Accordion("Negotiation Letter", open=False):
            email_btn = gr.Button("Draft pushback")
            email_out = gr.Textbox(lines=12, label="Draft for review", show_copy_button=True)

        def load_example(name):
            payload = get_example_payload(name)
            return payload["text"], payload["source_banner_html"]

        def load_file(path):
            if not path:
                return "", ""
            with open(path, "r", errors="ignore") as fh:
                return fh.read(), ""

        def analyze_blocks(text):
            data = analyze_contract_payload(text)
            return _render_blocks_results(data), data.get("highlighted_html", ""), json.dumps(data.get("findings", []))

        def draft_blocks(state_json):
            return draft_email_payload(state_json).get("email", "")

        ex.change(load_example, ex, [inp, src_banner])
        up.upload(load_file, up, [inp, src_banner])
        btn.click(analyze_blocks, inp, [out_cards, out_doc, st])
        email_btn.click(draft_blocks, st, email_out)
        clear_btn.click(lambda: ("", "", "", "", "[]"), None, [inp, src_banner, out_cards, out_doc, st])

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
