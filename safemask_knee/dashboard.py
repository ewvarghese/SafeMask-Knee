"""Read-only aggregate dashboard: never serve raw input, weight or ID files."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from .constants import DISCLAIMER

LOG = logging.getLogger(__name__)

# Entire page is self-contained: no CDN, tracking, remote JS or image requests.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeMask Knee · Research audit</title>
<style>
:root{color-scheme:dark;--bg:#08131b;--panel:#10232d;--edge:#2b4551;--text:#e8f2f2;--muted:#acc2c7;--accent:#79dac8;--caution:#f9ca81}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 80% -20%,#1a3b40,var(--bg) 65%);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text)}
main{max-width:1150px;margin:auto;padding:42px 24px 80px}h1{font-size:clamp(2rem,5vw,3.3rem);margin:8px 0 6px;letter-spacing:-.04em;line-height:1.1}h2{font-size:1.3rem;margin:0 0 14px;letter-spacing:-.02em}p{margin:8px 0 18px}.eyebrow{text-transform:uppercase;letter-spacing:.2em;font-size:.72rem;color:var(--accent);font-weight:800}.lead{color:var(--muted);max-width:820px}.badge{display:inline-block;color:var(--accent);background:#15423e;border:1px solid #28635e;border-radius:999px;padding:5px 12px;font-size:.76rem;font-weight:800;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:28px 0}.card,.section{background:linear-gradient(155deg,#132b34,var(--panel));border:1px solid var(--edge);border-radius:16px;padding:22px;box-shadow:0 12px 30px #0002}.card span{color:var(--muted);font-size:.82rem}.card strong{font-size:1.9rem;display:block;letter-spacing:-.04em}.section{margin-top:18px}.section p,.hint{color:var(--muted)}.caution{border-color:#806343;background:linear-gradient(135deg,#30291e,#181d23);color:var(--caution);padding:18px 22px;border-radius:14px;margin:20px 0}.caution p{color:#f4deb9;margin:0}.list{display:grid;gap:12px;margin-top:22px}.line{display:grid;grid-template-columns:180px 1fr 65px;gap:14px;align-items:center;font-size:.88rem}.line .name{font-weight:650}.track{height:12px;border-radius:20px;background:#203943;overflow:hidden}.fill{height:100%;border-radius:20px;background:linear-gradient(90deg,#3faaa4,#a5edce)}.line .value{text-align:right;font-variant-numeric:tabular-nums}.scroller{overflow:auto}table{width:100%;border-collapse:collapse;font-size:.88rem;min-width:580px}th,td{padding:10px 12px;border-bottom:1px solid var(--edge);text-align:left}th{color:var(--muted);font-weight:650}td{font-variant-numeric:tabular-nums}footer{margin-top:34px;color:var(--muted);font-size:.83rem;border-top:1px solid var(--edge);padding-top:18px}a{color:var(--accent)}@media(max-width:760px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.line{grid-template-columns:135px 1fr 50px;font-size:.77rem}}@media(max-width:420px){.grid{grid-template-columns:1fr}.line{grid-template-columns:110px 1fr 50px}}
</style></head><body><main>
<div class="eyebrow">SafeMask Knee · Research tools</div><h1>Selection audit</h1>
<p class="lead">Aggregate acceptance and weight stability across eleven knee findings. This display is read-only and does not expose study identifiers, labels or source CSVs.</p>
<div class="badge" id="status">Loading aggregate results…</div>
<div class="grid"><div class="card"><span>Studies</span><strong id="studies">—</strong></div><div class="card"><span>Finding pairs</span><strong id="pairs">—</strong></div><div class="card"><span>Accepted pairs</span><strong id="accepted">—</strong></div><div class="card"><span>Effective sample size</span><strong id="ess">—</strong></div></div>
<div class="caution"><strong>What is not measured</strong><p id="warning">Waiting for the input-quality and positivity diagnostics.</p></div>
<div class="section"><h2>Acceptance by finding</h2><p>Selected by the source confidence mask; acceptance is not correctness or disease prevalence.</p><div id="bars" class="list"></div></div>
<div class="section"><h2>IPCW sensitivity</h2><p>Effective sample size across propensity floors and weight caps. These variants do not establish bias correction.</p><div class="scroller"><table><thead><tr><th>Propensity floor</th><th>Weight cap</th><th>ESS</th><th>At cap</th><th>Configured</th></tr></thead><tbody id="sensitivity"></tbody></table></div></div>
<div class="section"><h2>Operational acceptance strata</h2><p>Shown only for groups with at least five studies. These are not protected-group fairness findings.</p><div class="scroller"><table><thead><tr><th>Attribute</th><th>Stratum</th><th>Studies</th><th>Acceptance</th></tr></thead><tbody id="strata"></tbody></table></div></div>
<footer><strong>For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.</strong><br>Only aggregate JSON is available at <code>/api/summary</code>. No clinical masks or image model can be inferred from this dashboard.</footer>
</main><script>
const n=x=>Number(x).toLocaleString();const pct=x=>(Number(x)*100).toFixed(1)+'%';
function cell(row,value){const td=document.createElement('td');td.textContent=String(value);row.append(td)}
fetch('/api/summary',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Aggregate results unavailable');return r.json()}).then(s=>{
  document.getElementById('status').textContent=s.expert_truth_available?'Expert truth present · interpret with caution':'Expert truth unavailable · label-free audit only';
  for(const [key,value] of Object.entries({studies:n(s.n_studies),pairs:n(s.n_pairs),accepted:n(s.n_accepted),ess:n(Math.round(s.effective_sample_size))}))document.getElementById(key).textContent=value;
  document.getElementById('warning').textContent=(s.expert_truth_available?'Expert-dependent results require independent validation. ':'No official per-study expert truth: accuracy, MNAR and risk masks are unavailable. ')+'OOF propensity clipping: '+pct(s.propensity_lower_clip_fraction)+' at floor; '+pct(s.propensity_upper_clip_fraction)+' at ceiling. Deterministic selection can violate positivity.';
  const bars=document.getElementById('bars');for(const f of s.findings){const line=document.createElement('div');line.className='line';const name=document.createElement('span');name.className='name';name.textContent=f.finding;const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=Math.max(0,Math.min(100,Number(f.acceptance_rate)*100))+'%';track.append(fill);const value=document.createElement('span');value.className='value';value.textContent=pct(f.acceptance_rate);line.append(name,track,value);bars.append(line)}
  const sensitivity=document.getElementById('sensitivity');for(const r of s.sensitivity){const tr=document.createElement('tr');for(const v of [Number(r.propensity_floor).toFixed(2),Number(r.weight_cap).toFixed(0),n(Math.round(r.effective_sample_size)),pct(r.fraction_at_cap),r.is_configured_baseline?'Yes':'—'])cell(tr,v);sensitivity.append(tr)}
  const strata=document.getElementById('strata');for(const r of s.operational_strata){const tr=document.createElement('tr');for(const v of [r.attribute,r.stratum,n(r.n_studies),pct(r.acceptance_rate)])cell(tr,v);strata.append(tr)}
}).catch(e=>{document.getElementById('status').textContent='Could not load aggregated results';document.getElementById('warning').textContent=e.message});
</script></body></html>"""


def dashboard_snapshot(root: Path) -> dict[str, object]:
    """Read *only* aggregate result files, never an input or patient-level CSV."""
    root = Path(root)
    required = (root / "quality/data_quality.json", root / "propensity_diagnostics.json",
                root / "sensitivity/sensitivity_overview.csv")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Run `train` first; dashboard aggregate is missing: {path}")
    quality = json.loads(required[0].read_text(encoding="utf-8"))
    diagnostics = json.loads(required[1].read_text(encoding="utf-8"))["summary"]
    sensitivity = pd.read_csv(required[2]).loc[:, [
        "propensity_floor", "weight_cap", "effective_sample_size",
        "fraction_at_cap", "is_configured_baseline"]]
    strata = []
    fairness = root / "audit/fairness_table.csv"
    if fairness.is_file():
        permitted = {"report_script_bucket", "report_length_bucket",
                     "multi_report_context_flag", "verified_mri_teacher_available",
                     "official_expert_evaluation_row"}
        subset = pd.read_csv(fairness)
        subset = subset.loc[subset.attribute.isin(permitted) & subset.n_studies.ge(5)]
        strata = subset.loc[:, ["attribute", "stratum", "n_studies",
                                "acceptance_rate"]].fillna("").to_dict("records")
    return {"disclaimer": DISCLAIMER, "status": "aggregate_only_read_only",
            "n_studies": quality["n_studies"], "n_pairs": quality["n_pairs"],
            "n_accepted": diagnostics["n_accepted"],
            "effective_sample_size": diagnostics["effective_sample_size_accepted"],
            "expert_truth_available": diagnostics["expert_truth_available"],
            "propensity_lower_clip_fraction": diagnostics["propensity_lower_clip_fraction"],
            "propensity_upper_clip_fraction": diagnostics["propensity_upper_clip_fraction"],
            "findings": [{"finding": r["finding"], "acceptance_rate": r["acceptance_rate"]}
                         for r in quality["finding_quality"]],
            "sensitivity": sensitivity.to_dict("records"),
            "operational_strata": strata}


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    """Build a strict allowlist-only HTTP handler with no static file serving."""
    resolved = Path(root).resolve()

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, media: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; "
                             "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "connect-src 'self'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib's callback spelling
            route = urlsplit(self.path).path
            if route == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/api/summary":
                try:
                    body = json.dumps(dashboard_snapshot(resolved), allow_nan=False).encode()
                except (FileNotFoundError, ValueError, KeyError) as exc:
                    LOG.error("Could not load aggregate dashboard: %s", exc)
                    self._send(503, b'{"error":"aggregate results unavailable"}',
                               "application/json; charset=utf-8")
                else:
                    self._send(200, body, "application/json; charset=utf-8")
            elif route == "/healthz":
                self._send(200, b'{"status":"ok"}', "application/json; charset=utf-8")
            else:
                self._send(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            self._send(405, b"Read-only dashboard", "text/plain; charset=utf-8")

        do_PUT = do_DELETE = do_PATCH = do_POST

    return DashboardHandler


def serve_dashboard(root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run an intentionally read-only aggregate HTTP preview until interrupted."""
    dashboard_snapshot(root)  # fail before opening a port if train has not run
    server = ThreadingHTTPServer((host, port), make_handler(root))
    LOG.info("Read-only aggregate dashboard at http://%s:%d/ (no raw files served)",
             host, server.server_address[1])
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
