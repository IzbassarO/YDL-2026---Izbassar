"""Export a SELF-CONTAINED *tutorial* search page (single HTML file).

This is not a product UI — it's a guide. For every query it shows the pipeline
the system runs (tokenize -> vectorize -> cosine -> rank), why each line matched
(highlighted words / shared trigrams / the 'concept' the query formed), and below
the search there are explainer sections on the data and the three methods.

Modes: Meaning (GloVe) · Keywords (tf-idf) · Fuzzy (char-trigram). All in-browser.
"""
import re
import json
import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

N_LINES = 5000
N_FREQUENT = 6000
TOK = re.compile(r"[a-z']+")

lines = pd.read_parquet("data/lines.parquet").sample(N_LINES, random_state=7).reset_index(drop=True)
wv = api.load("glove-wiki-gigaword-100")


def embed(text):
    vecs = [wv[w] for w in TOK.findall(text.lower()) if w in wv]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v / max(np.linalg.norm(v), 1e-9)


linevecs, keep = [], []
for i, line in enumerate(lines["line"]):
    v = embed(line)
    if v is None:
        continue
    linevecs.append([round(float(x), 3) for x in v])
    keep.append(i)
lines = lines.iloc[keep].reset_index(drop=True)
meta = [{"l": r.line, "a": r.artist, "s": r.song} for r in lines.itertuples()]

tfidf = TfidfVectorizer(stop_words="english", min_df=2)
tfidf.fit(lines["line"])
vocab = tfidf.vocabulary_
idf = {w: round(float(tfidf.idf_[j]), 3) for w, j in vocab.items()}
toks = [[w for w in TOK.findall(t.lower()) if w in vocab] for t in lines["line"]]

gvocab = set()
for line in lines["line"]:
    gvocab.update(w for w in TOK.findall(line.lower()) if w in wv)
gvocab.update(wv.index_to_key[:N_FREQUENT])
glove = {w: [round(float(x), 3) for x in wv[w]] for w in gvocab}

print(f"lines: {len(meta):,}  glove-vocab: {len(glove):,}  idf-vocab: {len(idf):,}")

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How Lyric Search Works — a guide</title>
<style>
 :root{color-scheme:light}
 *{box-sizing:border-box}
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#f6f7f9;color:#1f2328;
      max-width:840px;margin:0 auto;padding:40px 22px 96px;line-height:1.62}
 h1{font-size:29px;margin:0 0 6px;letter-spacing:-.02em;font-weight:700}
 h2{font-size:20px;margin:42px 0 12px;padding-bottom:8px;border-bottom:1px solid #e3e6ea;font-weight:650}
 h3{font-size:15px;margin:18px 0 6px;color:#1f2328;font-weight:650}
 p.sub{color:#57606a;margin:0 0 24px;font-size:15px;max-width:680px}
 .modes{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
 .mode{padding:9px 15px;border:1px solid #d4d8dd;border-radius:10px;background:#fff;color:#57606a;cursor:pointer;font-size:14px;transition:.15s}
 .mode:hover{border-color:#9aa4af}
 .mode.on{background:#3257d6;border-color:#3257d6;color:#fff;box-shadow:0 1px 3px rgba(50,87,214,.25)}
 .mode small{display:block;font-size:11px;opacity:.85;margin-top:1px}
 .box{display:flex;gap:10px} input{flex:1;padding:14px 16px;border-radius:10px;border:1px solid #d4d8dd;
      background:#fff;color:#1f2328;font-size:16px}
 input:focus{outline:none;border-color:#3257d6;box-shadow:0 0 0 3px rgba(50,87,214,.14)}
 button{padding:14px 22px;border:0;border-radius:10px;background:#3257d6;color:#fff;font-size:15px;cursor:pointer;font-weight:600}
 button:hover{background:#2746b8}
 .chips{margin:14px 0 22px}.chip{display:inline-block;background:#fff;border:1px solid #d4d8dd;color:#57606a;
      padding:6px 12px;border-radius:16px;margin:4px 6px 0 0;font-size:13px;cursor:pointer}
 .chip:hover{border-color:#9aa4af;color:#1f2328}
 .think{background:#eef2fd;border:1px solid #cfd9f6;border-radius:12px;padding:16px 18px;margin:8px 0 24px}
 .think h3{margin:0 0 10px;color:#3257d6;font-size:13px;text-transform:uppercase;letter-spacing:.05em;font-weight:700}
 .step{margin:8px 0;font-size:14px}.step b{color:#2746b8}
 .pill{display:inline-block;background:#fff;border:1px solid #d4d8dd;border-radius:6px;padding:1px 8px;margin:2px 3px;font-size:13px}
 .pill.hit{border-color:#3aa657;color:#1a7f37;background:#eaf7ee}.pill.miss{opacity:.55;text-decoration:line-through}
 .hit{padding:13px 15px;border:1px solid #e3e6ea;border-radius:10px;margin-bottom:10px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
 .hit .ly{font-size:17px;margin-bottom:3px;color:#1f2328}.hit .mt{color:#6a7178;font-size:13px}
 .hit .why{color:#8a9099;font-size:12px;margin-top:6px}
 .sc{float:right;color:#1a7f37;font-variant-numeric:tabular-nums;font-size:13px;font-weight:600}
 mark{background:#fff1bf;color:#7a5b00;border-radius:3px;padding:0 2px}
 .card{background:#fff;border:1px solid #e3e6ea;border-radius:12px;padding:18px 20px;margin:12px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
 code{background:#f0f2f5;border:1px solid #e3e6ea;border-radius:5px;padding:1px 6px;font-size:13px}
 table{width:100%;border-collapse:collapse;font-size:14px;margin:10px 0}
 th,td{border:1px solid #e3e6ea;padding:8px 10px;text-align:left}th{color:#57606a;font-weight:600;background:#f6f7f9}
 .muted{color:#6a7178;font-size:13px}
</style></head><body>

<h1>How Lyric Search Works</h1>
<p class="sub">An interactive guide. Type a query, pick a method, and follow how the system turns your words into a
ranked answer. Everything runs in your browser over __N__ lyric lines.</p>

<div class="modes" id="modes">
  <div class="mode on" data-m="glove">Meaning<small>GloVe · synonyms</small></div>
  <div class="mode" data-m="tfidf">Keywords<small>tf-idf · exact words</small></div>
  <div class="mode" data-m="tri">Fuzzy<small>trigram · typo-proof</small></div>
</div>
<div class="box"><input id="q" placeholder="e.g. feeling heartbroken and alone" autofocus>
<button onclick="run()">Search</button></div>
<div class="chips" id="chips"></div>

<div class="think" id="think"></div>
<h3 style="margin-top:0">Results <span class="muted" id="rescount"></span></h3>
<div id="out"></div>

<h2>What is this, and how does it work?</h2>
<div class="card">
<p><b>The goal.</b> Find lyric lines that match what you <i>mean</i>, not just the exact words you typed.
We take ~57,000 English songs (Spotify Million Song Dataset), split every song into single lines, and make each
line searchable. This page uses a 5,000-line sample so everything fits in one file.</p>
<p><b>The trick.</b> Turn every line into a <b>vector</b> (a list of numbers), turn your query into a vector the same way,
and rank lines by <b>cosine similarity</b> — the angle between the two vectors. Small angle = similar. That single idea
powers all three modes; they only differ in <i>how</i> they build the vector.</p>
</div>

<h2>The three methods</h2>

<h3>1 · Meaning — averaged GloVe embeddings</h3>
<div class="card">
<p><b>What it is.</b> GloVe is a set of pretrained word vectors (100 numbers per word) learned from huge text, so that
words used in similar contexts get similar vectors — the <i>distributional hypothesis</i>. <code>sad</code>, <code>lonely</code>
and <code>heartbroken</code> end up close together.</p>
<p><b>How a line becomes a vector.</b> Average the GloVe vectors of its words → one 100-d "meaning vector". Your query is
embedded the same way; we rank lines by cosine. This is why it finds <i>synonyms</i> with no shared words.</p>
<p class="muted">Trade-off: averaging throws away word order (like bag-of-words), and it can't embed misspelled words.</p>
</div>

<h3>2 · Keywords — tf-idf</h3>
<div class="card">
<p><b>What it is.</b> Classic bag-of-words. Each line is a sparse vector over the vocabulary; a word's weight is
<b>tf-idf</b> = how often it appears here × how <i>rare</i> it is overall (idf). Rare, specific words (a name, a place)
count more than common ones; stop-words are dropped.</p>
<p><b>How results come out.</b> Cosine between the query's tf-idf vector and each line's. Strong when you remember the exact
words; blind to synonyms and typos.</p>
</div>

<h3>3 · Fuzzy — character trigrams</h3>
<div class="card">
<p><b>What it is.</b> Break text into overlapping 3-letter chunks: <code>london → lon · ond · ndo · don</code>.
Two strings are similar if they share many chunks. A typo only changes a few chunks, so most still match.</p>
<p><b>How results come out.</b> We compare the chunk-sets of your query and each line (Dice overlap). This is the only mode
that survives <code>heartbrokn</code> → it still shares <code>hea·ear·art·rtb·bro·rok</code> with <code>heartbroken</code>.</p>
<p class="muted">Trade-off: it matches on spelling, not meaning — it has no idea what the words mean.</p>
</div>

<h2>How a single score is computed</h2>
<div class="card">
<p>Every mode ends the same way: <b>cosine similarity</b> between two vectors <code>a</code> and <code>b</code>:</p>
<p style="text-align:center"><code>cos(a, b) = (a · b) / (‖a‖ · ‖b‖)</code></p>
<p>It ranges 0 → 1 here (higher = closer). We sort all lines by this number and show the top 8. The score badge on each
result <i>is</i> this cosine. Try the same query in different modes and compare the scores and the winners.</p>
</div>

<h2>The data pipeline</h2>
<table>
<tr><th>Step</th><th>What happens</th><th>File</th></tr>
<tr><td>1. Load</td><td>57,650 songs (artist, title, lyrics)</td><td>spotify_millsongdata.csv</td></tr>
<tr><td>2. Split</td><td>lyrics → lines, clean, dedup → 180k lines</td><td>preprocess.py</td></tr>
<tr><td>3. Index</td><td>tf-idf + averaged GloVe per line</td><td>build_index.py</td></tr>
<tr><td>4. Index</td><td>character-trigram tf-idf per line</td><td>build_trigram.py</td></tr>
<tr><td>5. Ship</td><td>bake 5k lines + vectors into this page</td><td>export_web.py</td></tr>
</table>
<p class="muted">Techniques only from the Day-2 lab: tokenization, tf-idf, pretrained word embeddings, cosine similarity,
character n-grams. No BERT, no large language models.</p>

<script>
const GLOVE=__GLOVE__, META=__META__, VECS=__VECS__, IDF=__IDF__, TOKS=__TOKS__, DIM=100;
let MODE="glove";

const lineNorm=TOKS.map(ts=>{let s=0;const seen=new Set();
  for(const w of ts){if(!seen.has(w)){seen.add(w);const d=IDF[w]||0;s+=d*d}}return Math.sqrt(s)||1});
function trigrams(str){ // per-word char 3-grams, within word boundaries (matches sklearn char_wb)
  const words=str.toLowerCase().match(/[a-z']+/g)||[];const set=new Set();
  for(const w of words){const p=" "+w+" ";for(let i=0;i<p.length-2;i++)set.add(p.slice(i,i+3));}return set;}
const lineTri=META.map(m=>trigrams(m.l));
const GWORDS=Object.keys(GLOVE), GNORM={};
for(const w of GWORDS){let s=0;const v=GLOVE[w];for(let i=0;i<DIM;i++)s+=v[i]*v[i];GNORM[w]=Math.sqrt(s)||1;}

function toks(q){return (q.toLowerCase().match(/[a-z']+/g)||[]);}
function embed(text){const ts=toks(text);const v=new Float32Array(DIM);let n=0;
  for(const t of ts){const g=GLOVE[t];if(g){for(let i=0;i<DIM;i++)v[i]+=g[i];n++}}
  if(!n)return null;let s=0;for(let i=0;i<DIM;i++){v[i]/=n;s+=v[i]*v[i]}
  s=Math.sqrt(s)||1;for(let i=0;i<DIM;i++)v[i]/=s;return v;}
function nearestWords(qv,k){const out=[];
  for(const w of GWORDS){const g=GLOVE[w];let d=0;for(let i=0;i<DIM;i++)d+=g[i]*qv[i];out.push([d/GNORM[w],w])}
  out.sort((a,b)=>b[0]-a[0]);return out.slice(0,k).map(x=>x[1]);}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function hl(line,words){let h=esc(line);for(const w of words){if(!w)continue;
  h=h.replace(new RegExp('\\b('+w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')\\b','ig'),'<mark>$1</mark>')}return h;}

function setThink(html){document.getElementById('think').innerHTML=html;}
function pills(arr,cls){return arr.map(x=>`<span class="pill ${cls||''}">${esc(x)}</span>`).join('');}

function run(){
  const q=document.getElementById('q').value.trim();
  const out=document.getElementById('out');const cnt=document.getElementById('rescount');
  if(!q){setThink('<h3>How the system thinks</h3><div class="step">Type a query above to see the pipeline.</div>');out.innerHTML='';cnt.textContent='';return;}
  const qt=toks(q);
  let scored=null, detail=null, think='';

  if(MODE==='glove'){
    const found=qt.filter(t=>GLOVE[t]), miss=qt.filter(t=>!GLOVE[t]);
    const qv=embed(q);
    think=`<h3>How the system thinks · Meaning</h3>
      <div class="step"><b>1. Tokenize</b> → ${pills(qt)}</div>
      <div class="step"><b>2. Look up GloVe vectors</b> → found ${found.length}/${qt.length} ${miss.length?'· skipped '+pills(miss,'miss'):''}</div>
      <div class="step"><b>3. Average</b> the ${found.length} word-vectors → one 100-d <i>meaning vector</i></div>`;
    if(qv){think+=`<div class="step"><b>4. The query reads as</b> → ${pills(nearestWords(qv,6),'hit')} <span class="muted">(nearest words to that vector)</span></div>
      <div class="step"><b>5. Rank</b> all ${META.length} lines by cosine to it.</div>`;
      scored=VECS.map((vec,i)=>{let d=0;for(let k=0;k<DIM;k++)d+=vec[k]*qv[k];return[d,i]});}
    else think+=`<div class="step" style="color:#d1242f">No known words to embed — try the Fuzzy mode for typos.</div>`;
  }
  else if(MODE==='tfidf'){
    const kept=qt.filter(t=>IDF[t]), dropped=qt.filter(t=>!IDF[t]);
    think=`<h3>How the system thinks · Keywords</h3>
      <div class="step"><b>1. Tokenize</b> → ${pills(qt)}</div>
      <div class="step"><b>2. Keep words in vocab</b> (drop stop-words / unknown) → ${pills(kept,'hit')} ${dropped.length?'· dropped '+pills(dropped,'miss'):''}</div>
      <div class="step"><b>3. Weight by idf</b> (rarer = heavier): ${kept.map(w=>`${esc(w)}=<b>${IDF[w]}</b>`).join(' · ')||'—'}</div>
      <div class="step"><b>4. Rank</b> lines by tf-idf cosine; matched words get <mark>highlighted</mark>.</div>`;
    if(kept.length){let qn=0;const qw={};for(const w of kept){qw[w]=IDF[w];qn+=IDF[w]*IDF[w]}qn=Math.sqrt(qn)||1;
      scored=TOKS.map((ts,i)=>{let d=0;const seen=new Set();
        for(const w of ts){if(qw[w]&&!seen.has(w)){seen.add(w);d+=qw[w]*IDF[w]}}return[d/(lineNorm[i]*qn),i]});}
    detail={type:'words',words:kept};
  }
  else{
    const qs=trigrams(q);
    think=`<h3>How the system thinks · Fuzzy</h3>
      <div class="step"><b>1. Split each word</b> into 3-letter chunks → ${pills([...qs].map(x=>x.replace(/ /g,'␣')).slice(0,16))}${qs.size>16?' …':''}</div>
      <div class="step"><b>2. Split every line</b> into chunks the same way (per word)</div>
      <div class="step"><b>3. Score</b> by shared chunks (Dice overlap) — typos still share most chunks, so they match.</div>`;
    scored=lineTri.map((ls,i)=>{let inter=0;for(const g of qs)if(ls.has(g))inter++;return[2*inter/(qs.size+ls.size||1),i]});
    detail={type:'tri',qs:qs};
  }

  setThink(think);
  if(!scored){out.innerHTML='';cnt.textContent='';return;}
  scored.sort((a,b)=>b[0]-a[0]);
  const top=scored.slice(0,8).filter(s=>s[0]>0);
  cnt.textContent=top.length?`· top ${top.length} by cosine`:'';
  if(!top.length){out.innerHTML='<p class="muted">No matches in this mode — try another mode or other words.</p>';return;}
  out.innerHTML=top.map(([sc,i])=>{const m=META[i];
    let line=esc(m.l), why='';
    if(detail&&detail.type==='words'){line=hl(m.l,detail.words);
      const matched=detail.words.filter(w=>new RegExp('\\b'+w+'\\b','i').test(m.l));
      why=matched.length?`matched word${matched.length>1?'s':''}: ${matched.join(', ')}`:'';}
    else if(detail&&detail.type==='tri'){const ls=lineTri[i];const sh=[...detail.qs].filter(g=>ls.has(g));
      why=`shared chunks: ${sh.slice(0,8).map(x=>x.trim()||'␣').join(' · ')}${sh.length>8?' …':''}`;}
    return `<div class="hit"><span class="sc">${sc.toFixed(2)}</span>
      <div class="ly">"${line}"</div><div class="mt">${esc(m.a)} — ${esc(m.s)}</div>
      ${why?`<div class="why">↳ ${esc(why)}</div>`:''}</div>`}).join('');
}

document.querySelectorAll('.mode').forEach(el=>el.onclick=()=>{
  document.querySelectorAll('.mode').forEach(e=>e.classList.remove('on'));
  el.classList.add('on');MODE=el.dataset.m;run();});
const examples=["feeling heartbroken and alone","heartbrokn and lonley (typo)","i want to dance all night",
                "missing my hometown","money and power"];
const chips=document.getElementById('chips');
examples.forEach(e=>{const c=document.createElement('span');c.className='chip';c.textContent=e;
  c.onclick=()=>{document.getElementById('q').value=e.replace(' (typo)','');run()};chips.appendChild(c)});
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')run()});
run();
</script></body></html>"""

html = (TEMPLATE.replace("__GLOVE__", json.dumps(glove))
                .replace("__META__", json.dumps(meta, ensure_ascii=False))
                .replace("__VECS__", json.dumps(linevecs))
                .replace("__IDF__", json.dumps(idf))
                .replace("__TOKS__", json.dumps(toks))
                .replace("__N__", f"{len(meta):,}"))
with open("artifacts/search.html", "w") as f:
    f.write(html)
print("saved -> artifacts/search.html (tutorial)")
