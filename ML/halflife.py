#!/usr/bin/env python3
"""
Knowledge Half-Life of medRxiv LLM-evaluation preprints.

Metric (same as v1, obsolescence-ratio):
    reign H = next_same_vendor_frontier_release - evaluated_model_release
    age     = publication_date - evaluated_model_release
    R       = age / H                  (obsolescence ratio; >1 = stale on arrival)
    relevance_at_pub = 0.5 ** R        (exp. decay, half-life = one model reign)
Outputs: ai_eval_halflife_medrxiv.csv / .json  and  dashboard.html (self-contained).
"""
import csv, json
from datetime import date

TODAY = date(2026, 6, 15)

# Frontier model release anchors (verified from vendor / tracker timelines)
RELEASES = {
    "GPT-3.5 Turbo":     date(2022, 11, 30),
    "GPT-4":             date(2023, 3, 14),
    "GPT-4 Turbo":       date(2023, 11, 6),
    "GPT-4o":            date(2024, 5, 13),
    "o1":                date(2024, 9, 12),
    "GPT-4.5":           date(2025, 2, 27),
    "GPT-5":             date(2025, 8, 7),
    "GPT-5.1":           date(2025, 11, 12),
    "GPT-5.2":           date(2025, 12, 11),
    "Claude 3.7 Sonnet": date(2025, 2, 24),
    "Claude 4":          date(2025, 5, 22),
    "Claude Sonnet 4.5": date(2025, 9, 29),
    "Claude Opus 4.5":   date(2025, 11, 24),
}

# Successor = next same-vendor frontier model of equal-or-higher capability tier
SUCCESSOR = {
    "GPT-4":             "GPT-4 Turbo",
    "GPT-4o":            "o1",
    "o1":                "GPT-4.5",
    "GPT-5":             "GPT-5.1",
    "GPT-5.1":           "GPT-5.2",
    "Claude Sonnet 4.5": "Claude Opus 4.5",
}

# 21 real medRxiv preprints (DOI = verifiable link). pub = medRxiv posting date.
PAPERS = [
    dict(id="M01", model="GPT-4", pub=date(2024,3,12), field="Diagnostic reasoning (RCT)",
         title="Goh et al. — Influence of an LLM on Diagnostic Reasoning: a randomized clinical trial",
         doi="10.1101/2024.03.12.24303785"),
    dict(id="M02", model="GPT-4", pub=date(2024,6,24), field="Clinical information retrieval",
         title="Customizing GPT-4 for clinical information retrieval from hospital SOPs (vs Claude-3-Opus)",
         doi="10.1101/2024.06.24.24309221"),
    dict(id="M03", model="GPT-4", pub=date(2024,5,1), field="Evidence synthesis",
         title="Evaluating an LLM's ability to answer clinicians' questions vs librarian gold-standard",
         doi="10.1101/2024.05.01.24306691"),
    dict(id="M04", model="GPT-4", pub=date(2024,5,17), field="Biomedical NLP benchmark",
         title="Evaluation of LLM performance on the Biomedical Language Understanding (BLURB) benchmark",
         doi="10.1101/2024.05.17.24307411"),
    dict(id="M05", model="GPT-4", pub=date(2024,4,3), field="Clinical documentation (ED)",
         title="Evaluating LLMs for drafting Emergency Department discharge summaries",
         doi="10.1101/2024.04.03.24305088"),
    dict(id="M06", model="GPT-4o", pub=date(2025,10,19), field="Ophthalmology / safety",
         title="Recognizing 'Conformity Bias' in LLMs: a new risk for clinical use (ocular disease)",
         doi="10.1101/2025.10.19.25338293"),
    dict(id="M07", model="GPT-4o", pub=date(2025,12,5), field="Public-health data analytics (code)",
         title="Evaluating LLMs for natural-language-to-code in public health analytics (Czech)",
         doi="10.64898/2025.12.05.25341697"),
    dict(id="M08", model="GPT-4o", pub=date(2025,4,25), field="Pediatrics",
         title="Are LLMs ready for pediatrics? Comparative MedQA accuracy across clinical domains",
         doi="10.1101/2025.04.25.25326437"),
    dict(id="M09", model="GPT-5", pub=date(2025,9,10), field="Medical licensing (Japan)",
         title="Textbook-level medical knowledge in LLMs on the Japanese National Medical Exam",
         doi="10.1101/2025.09.10.25335398"),
    dict(id="M10", model="GPT-5", pub=date(2026,2,18), field="Clinical genetics",
         title="Performance characteristics of reasoning LLMs for ACMG/ClinGen PS4 variant curation",
         doi="10.64898/2026.02.18.26346543"),
    dict(id="M11", model="GPT-5.1", pub=date(2026,1,27), field="Medical reasoning benchmark",
         title="MedEvalArena: a self-generated, peer-judged benchmark for medical reasoning",
         doi="10.64898/2026.01.27.26344905"),
    dict(id="M12", model="Claude Sonnet 4.5", pub=date(2026,2,26), field="Medical AI safety (red-teaming)",
         title="Red-teaming Medical AI: systematic adversarial evaluation (Claude Sonnet 4.5)",
         doi="10.64898/2026.02.26.26347212"),
    dict(id="M13", model="GPT-5", pub=date(2025,9,9), field="Clinical prediction (ED revisit)",
         title="From clinical judgment to LLMs: predicting ED revisit/admission risk",
         doi="10.1101/2025.09.09.25335411"),
    dict(id="M14", model="o1", pub=date(2026,2,22), field="Diagnostic error correction",
         title="LLMs as a diagnostic safety net: challenging vs confirming physician misdiagnosis",
         doi="10.1101/2026.02.22.26346832"),
    dict(id="M15", model="GPT-4o", pub=date(2024,6,18), field="Medical education (USMLE vision)",
         title="Evaluating ChatGPT-4o vision capabilities on image-based USMLE questions",
         doi="10.1101/2024.06.18.24309092"),
    dict(id="M16", model="GPT-5", pub=date(2025,8,28), field="Medical licensing (USMLE)",
         title="A performance analysis of the GPT-5 family in medical question answering",
         doi="10.1101/2025.08.28.25334657"),
    dict(id="M17", model="GPT-4o", pub=date(2025,4,22), field="Neuro-oncology imaging",
         title="Artificial intelligence in neuro-oncology: assessing ChatGPT-4o on tumour imaging",
         doi="10.1101/2025.04.22.25326204"),
    dict(id="M18", model="GPT-4o", pub=date(2025,1,6), field="Mental health triage",
         title="Potential of ChatGPT in youth mental-health emergency triage",
         doi="10.1101/2025.01.06.24319771"),
    dict(id="M19", model="GPT-4", pub=date(2025,3,18), field="Radiology reporting (review)",
         title="LLMs in radiology reporting — a systematic review of performance and limitations",
         doi="10.1101/2025.03.18.25324193"),
    dict(id="M20", model="GPT-4o", pub=date(2025,7,23), field="Patient education (radiology)",
         title="Evaluation of LLM-generated patient information for radiation-risk communication",
         doi="10.1101/2025.07.23.25332093"),
    dict(id="M21", model="GPT-4", pub=date(2024,9,1), field="Cardiology (review)",
         title="Large language models in cardiology: a systematic review",
         doi="10.1101/2024.09.01.24312887"),
]

def days(a, b): return (b - a).days

def analyze(p):
    m = p["model"]; rel = RELEASES[m]; pub = p["pub"]
    sname = SUCCESSOR.get(m); sdate = RELEASES.get(sname) if sname else None
    age = days(rel, pub)
    if sdate:
        reign = max(days(rel, sdate), 1); known = True
    else:
        reign = max(days(rel, TODAY), 1); sname = "(none yet)"; known = False
    R = age / reign
    return dict(
        id=p["id"], title=p["title"], field=p["field"], model=m,
        doi=p["doi"], url=f"https://doi.org/{p['doi']}",
        model_release=rel.isoformat(), publication_date=pub.isoformat(),
        successor_model=sname, successor_release=sdate.isoformat() if sdate else "",
        age_at_pub_days=age, model_reign_days=reign, halflife_days=reign,
        obsolescence_ratio_R=round(R, 2), relevance_at_pub=round(0.5**R, 3),
        stale_on_arrival=bool(sdate and sdate <= pub),
        days_frontier_after_pub=(days(pub, sdate) if sdate else days(pub, TODAY)),
        successor_known=known,
    )

def main():
    rows = [analyze(p) for p in PAPERS]
    rows.sort(key=lambda r: r["obsolescence_ratio_R"], reverse=True)

    cols = ["id","title","field","model","model_release","publication_date",
            "successor_model","successor_release","age_at_pub_days","model_reign_days",
            "halflife_days","obsolescence_ratio_R","relevance_at_pub",
            "stale_on_arrival","days_frontier_after_pub","doi","url"]
    with open("ai_eval_halflife_medrxiv.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in cols})
    with open("ai_eval_halflife_medrxiv.json","w") as f:
        json.dump(rows, f, indent=2)

    n=len(rows); stale=sum(r["stale_on_arrival"] for r in rows)
    avgR=sum(r["obsolescence_ratio_R"] for r in rows)/n
    hl=sorted(r["halflife_days"] for r in rows)[n//2]
    stats=dict(n=n, stale=stale, stale_pct=round(100*stale/n), mean_R=round(avgR,2),
               median_halflife_days=hl, generated=TODAY.isoformat())
    print(stats)
    write_html(rows, stats)

def write_html(rows, stats):
    data = json.dumps(rows)
    st = json.dumps(stats)
    html = HTML_TEMPLATE.replace("__DATA__", data).replace("__STATS__", st)
    with open("dashboard.html","w") as f: f.write(html)
    print("wrote dashboard.html")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Half-Life of medRxiv LLM-Evaluation Preprints</title>
<style>
:root{--bg:#0f1420;--panel:#171e2e;--panel2:#1d2740;--ink:#e8edf7;--mut:#93a1bd;
--fresh:#2dd4a7;--mid:#f5c451;--stale:#ff6b6b;--line:#2a3756;--accent:#6aa6ff;}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0c111c,#0f1420);color:var(--ink);
font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:24px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--mut);margin:0 0 18px;max-width:900px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:150px}
.card .v{font-size:26px;font-weight:700}
.card .l{color:var(--mut);font-size:12px}
.layout{display:grid;grid-template-columns:1.05fr 1fr;gap:20px;align-items:start}
@media(max-width:980px){.layout{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
.panel h2{font-size:15px;margin:0 0 10px}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
select,input{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:7px 10px;font-size:13px}
input{min-width:200px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{padding:8px 9px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{position:sticky;top:0;background:var(--panel);cursor:pointer;user-select:none;white-space:nowrap;color:var(--mut)}
th:hover{color:var(--ink)}
th.num,td.num{text-align:right}
tbody tr:hover{background:#202b45}
.tag{display:inline-block;padding:2px 7px;border-radius:999px;font-size:11px;font-weight:600}
.t-fresh{background:rgba(45,212,167,.15);color:var(--fresh)}
.t-mid{background:rgba(245,196,81,.15);color:var(--mid)}
.t-stale{background:rgba(255,107,107,.15);color:var(--stale)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.bar{height:7px;border-radius:4px;background:var(--panel2);overflow:hidden;min-width:60px}
.bar>i{display:block;height:100%}
.tablewrap{max-height:620px;overflow:auto;border-radius:10px}
.legend{display:flex;gap:16px;color:var(--mut);font-size:12px;margin-top:8px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.note{color:var(--mut);font-size:11.5px;margin-top:10px}
#tip{position:fixed;pointer-events:none;background:#0a0f1a;border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font-size:12px;max-width:280px;opacity:0;transition:opacity .1s;z-index:9}
.muted{color:var(--mut)}
</style></head>
<body>
<h1>Knowledge Half-Life of medRxiv LLM-Evaluation Preprints</h1>
<p class="sub">Each preprint evaluates a named frontier model on a clinical task. Its relevance decays with a
half-life equal to the model's <b>reign</b> (time until the next same-vendor frontier model). The
<b>obsolescence ratio R</b> = (publication&nbsp;&minus;&nbsp;model&nbsp;release) / reign. R&nbsp;&ge;&nbsp;1 means the paper was
already evaluating a superseded model on the day it was posted. Click any column header to sort. Generated <span id="gen"></span>.</p>

<div class="cards" id="cards"></div>

<div class="layout">
  <div class="panel">
    <h2>Universal decay curve &amp; paper positions</h2>
    <div id="chart"></div>
    <div class="legend">
      <span><span class="dot" style="background:var(--fresh)"></span>fresh (R&lt;1)</span>
      <span><span class="dot" style="background:var(--mid)"></span>aging (1&le;R&lt;2)</span>
      <span><span class="dot" style="background:var(--stale)"></span>stale (R&ge;2)</span>
      <span class="muted">curve: relevance = 0.5<sup>R</sup></span>
    </div>
    <p class="note">X = obsolescence ratio R (how many model-reigns elapsed by publication). Y = retained
    relevance at publication. Hover a dot for the paper. Filters below update the chart too.</p>
  </div>

  <div class="panel">
    <h2>Papers <span class="muted" id="count"></span></h2>
    <div class="controls">
      <select id="fField"><option value="">All fields</option></select>
      <select id="fModel"><option value="">All models</option></select>
      <select id="fStale"><option value="">All</option><option value="1">Stale on arrival</option><option value="0">Fresh on arrival</option></select>
      <input id="fSearch" placeholder="Search title / field…">
    </div>
    <div class="tablewrap"><table id="tbl"><thead><tr>
      <th data-k="field">Field</th>
      <th data-k="title">Paper</th>
      <th data-k="model">Model</th>
      <th data-k="publication_date">Posted</th>
      <th data-k="halflife_days" class="num">Half-life (d)</th>
      <th data-k="obsolescence_ratio_R" class="num">R</th>
      <th data-k="relevance_at_pub" class="num">Relevance@pub</th>
    </tr></thead><tbody></tbody></table></div>
    <p class="note">Half-life = the evaluated model's reign in days (one half-life of the paper's headline
    claim). Links go to the medRxiv DOI. Reviews are included and labelled.</p>
  </div>
</div>

<div id="tip"></div>
<script>
const DATA=__DATA__, STATS=__STATS__;
document.getElementById('gen').textContent=STATS.generated;

const cards=[
 ['Preprints',STATS.n,''],
 ['Stale on arrival',STATS.stale+' ('+STATS.stale_pct+'%)','successor already out at posting'],
 ['Mean obsolescence R',STATS.mean_R,'avg reigns elapsed at posting'],
 ['Median half-life',STATS.median_halflife_days+' d','~'+(STATS.median_halflife_days/30.4).toFixed(1)+' months'],
];
document.getElementById('cards').innerHTML=cards.map(c=>
 `<div class="card"><div class="v">${c[1]}</div><div class="l">${c[0]}</div>${c[2]?`<div class="l">${c[2]}</div>`:''}</div>`).join('');

// populate filters
const fField=document.getElementById('fField'),fModel=document.getElementById('fModel');
[...new Set(DATA.map(d=>d.field))].sort().forEach(v=>fField.add(new Option(v,v)));
[...new Set(DATA.map(d=>d.model))].sort().forEach(v=>fModel.add(new Option(v,v)));

let sortKey='obsolescence_ratio_R',sortDir=-1;
function band(R){return R<1?'fresh':R<2?'mid':'stale';}
function col(R){return R<1?'var(--fresh)':R<2?'var(--mid)':'var(--stale)';}

function filtered(){
  const f=fField.value,m=fModel.value,s=document.getElementById('fStale').value,
        q=document.getElementById('fSearch').value.toLowerCase();
  return DATA.filter(d=>(!f||d.field===f)&&(!m||d.model===m)&&
    (s===''||String(d.stale_on_arrival?1:0)===s)&&
    (!q||(d.title+' '+d.field).toLowerCase().includes(q)));
}
function sorted(rows){
  return rows.slice().sort((a,b)=>{
    let x=a[sortKey],y=b[sortKey];
    if(typeof x==='string'){x=x.toLowerCase();y=y.toLowerCase();}
    return x<y?-1*sortDir:x>y?1*sortDir:0;});
}
function render(){
  const rows=sorted(filtered());
  document.getElementById('count').textContent='— '+rows.length+' shown';
  const maxHL=Math.max(...DATA.map(d=>d.halflife_days));
  document.querySelector('#tbl tbody').innerHTML=rows.map(d=>{
    const b=band(d.obsolescence_ratio_R);
    const tag=b==='fresh'?'t-fresh':b==='mid'?'t-mid':'t-stale';
    return `<tr>
      <td>${d.field}</td>
      <td><a href="${d.url}" target="_blank" rel="noopener">${d.title}</a>
          <div class="muted">${d.doi}${d.stale_on_arrival?' · superseded by '+d.successor_model:''}</div></td>
      <td>${d.model}<div class="muted">${d.model_release}</div></td>
      <td>${d.publication_date}</td>
      <td class="num">${d.halflife_days}<div class="bar" style="margin-top:3px"><i style="width:${100*d.halflife_days/maxHL}%;background:var(--accent)"></i></div></td>
      <td class="num"><span class="tag ${tag}">${d.obsolescence_ratio_R}</span></td>
      <td class="num">${(d.relevance_at_pub*100).toFixed(0)}%<div class="bar" style="margin-top:3px"><i style="width:${100*d.relevance_at_pub}%;background:${col(d.obsolescence_ratio_R)}"></i></div></td>
    </tr>`;}).join('');
  drawChart(rows);
}
document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=(k==='field'||k==='title'||k==='model'||k==='publication_date')?1:-1;}
  render();});
['fField','fModel','fStale','fSearch'].forEach(id=>document.getElementById(id).addEventListener('input',render));

// ---- SVG decay chart ----
const tip=document.getElementById('tip');
function drawChart(rows){
  const W=440,H=300,pad={l:46,r:14,t:14,b:38};
  const maxR=Math.max(2.2,Math.ceil(Math.max(...DATA.map(d=>d.obsolescence_ratio_R))*10)/10);
  const X=r=>pad.l+(r/maxR)*(W-pad.l-pad.r);
  const Y=v=>pad.t+(1-v)*(H-pad.t-pad.b);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:480px">`;
  // grid + y labels
  for(let i=0;i<=5;i++){const v=i/5,y=Y(v);
    s+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="#223" stroke-width="1"/>`;
    s+=`<text x="${pad.l-6}" y="${y+3}" fill="#7d8aa8" font-size="10" text-anchor="end">${(v*100).toFixed(0)}%</text>`;}
  // x ticks
  for(let i=0;i<=Math.round(maxR);i++){if(i>maxR)break;const x=X(i);
    s+=`<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H-pad.b}" stroke="#1b2740" stroke-width="1"/>`;
    s+=`<text x="${x}" y="${H-pad.b+15}" fill="#7d8aa8" font-size="10" text-anchor="middle">R=${i}</text>`;}
  // R=1 marker (superseded threshold)
  s+=`<line x1="${X(1)}" y1="${pad.t}" x2="${X(1)}" y2="${H-pad.b}" stroke="#ff6b6b" stroke-dasharray="4 3" stroke-width="1.2"/>`;
  s+=`<text x="${X(1)+4}" y="${pad.t+12}" fill="#ff8c8c" font-size="9">stale →</text>`;
  // decay curve y=0.5^R
  let path='';
  for(let i=0;i<=120;i++){const r=maxR*i/120;const x=X(r),y=Y(Math.pow(0.5,r));path+=(i?'L':'M')+x.toFixed(1)+' '+y.toFixed(1)+' ';}
  s+=`<path d="${path}" fill="none" stroke="#6aa6ff" stroke-width="2"/>`;
  // axis titles
  s+=`<text x="${pad.l+(W-pad.l-pad.r)/2}" y="${H-4}" fill="#93a1bd" font-size="10" text-anchor="middle">obsolescence ratio R (model-reigns elapsed at posting)</text>`;
  s+=`<text x="14" y="${pad.t+(H-pad.t-pad.b)/2}" fill="#93a1bd" font-size="10" text-anchor="middle" transform="rotate(-90 14 ${pad.t+(H-pad.t-pad.b)/2})">retained relevance</text>`;
  // points
  rows.forEach((d,idx)=>{const x=X(Math.min(d.obsolescence_ratio_R,maxR)),y=Y(d.relevance_at_pub);
    s+=`<circle cx="${x}" cy="${y}" r="5" fill="${col(d.obsolescence_ratio_R)}" stroke="#0c111c" stroke-width="1.2"
        data-i="${DATA.indexOf(d)}" class="pt" style="cursor:pointer"/>`;});
  s+=`</svg>`;
  document.getElementById('chart').innerHTML=s;
  document.querySelectorAll('.pt').forEach(c=>{
    c.onmousemove=e=>{const d=DATA[c.dataset.i];
      tip.innerHTML=`<b>${d.title}</b><br>${d.field} · ${d.model}<br>R=${d.obsolescence_ratio_R} · relevance ${(d.relevance_at_pub*100).toFixed(0)}% · half-life ${d.halflife_days}d`;
      tip.style.opacity=1;tip.style.left=Math.min(e.clientX+14,innerWidth-300)+'px';tip.style.top=(e.clientY+14)+'px';};
    c.onmouseleave=()=>tip.style.opacity=0;});
}
render();
</script>
</body></html>"""

if __name__=="__main__":
    main()
