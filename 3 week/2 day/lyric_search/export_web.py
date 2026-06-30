"""Export a SELF-CONTAINED tutorial search page (single HTML file), production-grade UI.

Three modes (switch in the UI): Meaning (GloVe) · Keywords (tf-idf) · Fuzzy (char-trigram).
A live side panel shows the pipeline the system runs for your query; below, explainer cards
teach each method. Everything (vectors, idf, tokens, text) is baked in and runs in the browser.
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
_w = json.load(open("data/word_idf.json"))
word_idf, default_w = _w["idf"], _w["default"]


def embed(text):
    """idf-weighted average of GloVe vectors (frequent fillers count less)."""
    num = np.zeros(wv.vector_size, dtype=np.float32)
    den = 0.0
    for w in TOK.findall(text.lower()):
        if w in wv:
            wt = word_idf.get(w, default_w)
            num += wt * wv[w]
            den += wt
    if den == 0:
        return None
    v = num / den
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
weight = {w: round(word_idf.get(w, default_w), 2) for w in gvocab}   # idf weight per shipped word

print(f"lines: {len(meta):,}  glove-vocab: {len(glove):,}  idf-vocab: {len(idf):,}")

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lyric Search — find a line by meaning</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
 :root{
   --bg:#f3f4f8; --card:#ffffff; --ink:#1b1d26; --muted:#6b7180; --faint:#9aa0ae;
   --line:#e7e9f0; --accent:#5457e6; --accent2:#8b5cf6; --accentSoft:#eef0fe; --ok:#1a7f37;
   color-scheme:light;
 }
 *{box-sizing:border-box}
 html{scroll-behavior:smooth}
 body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
      background:var(--bg);color:var(--ink);margin:0;line-height:1.6;-webkit-font-smoothing:antialiased}
 .wrap{max-width:1000px;margin:0 auto;padding:0 22px}

 /* hero */
 .hero{background:radial-gradient(1200px 300px at 50% -120px,rgba(99,102,241,.14),transparent),
       linear-gradient(180deg,#ffffff,var(--bg));border-bottom:1px solid var(--line);padding:46px 0 30px}
 .kicker{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
 h1{font-size:38px;line-height:1.1;letter-spacing:-.025em;margin:0 0 10px;font-weight:700}
 .tagline{color:var(--muted);font-size:17px;margin:0 0 24px;max-width:620px}
 .searchbar{display:flex;align-items:center;gap:8px;background:var(--card);border:1px solid #d8dbe6;
       border-radius:14px;padding:7px 8px 7px 16px;box-shadow:0 6px 22px rgba(28,30,60,.07);transition:.18s}
 .searchbar:focus-within{border-color:var(--accent);box-shadow:0 0 0 4px rgba(84,87,230,.14)}
 .searchbar svg{flex:0 0 20px;color:var(--faint)}
 .searchbar input{flex:1;border:0;outline:0;background:transparent;font:inherit;font-size:17px;padding:12px 6px;color:var(--ink)}
 .searchbar button{border:0;border-radius:10px;padding:12px 22px;font:inherit;font-weight:600;color:#fff;cursor:pointer;
       background:linear-gradient(135deg,var(--accent),var(--accent2));transition:.15s}
 .searchbar button:hover{filter:brightness(1.06);transform:translateY(-1px)}
 .segmented{display:inline-flex;background:#e8eaf2;border-radius:12px;padding:4px;margin-top:16px;gap:3px}
 .seg{padding:8px 16px;border:0;border-radius:9px;background:transparent;cursor:pointer;font:inherit;color:var(--muted);
       display:flex;flex-direction:column;line-height:1.25;text-align:center;transition:.15s}
 .seg small{font-size:11px;opacity:.85;font-weight:400}
 .seg.on{background:#fff;color:var(--ink);font-weight:600;box-shadow:0 1px 4px rgba(28,30,60,.14)}
 .chips{margin-top:16px}
 .chip{display:inline-block;background:#fff;border:1px solid #dadde7;color:var(--muted);padding:6px 13px;
       border-radius:18px;margin:0 7px 8px 0;font-size:13px;cursor:pointer;transition:.15s}
 .chip:hover{border-color:var(--accent);color:var(--accent);background:var(--accentSoft)}

 /* layout */
 .layout{display:grid;grid-template-columns:1.55fr 1fr;gap:26px;align-items:start;padding:28px 0 8px}
 .col-head{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin:0 0 12px}
 .think-col{position:sticky;top:18px}

 /* results */
 .hit{display:flex;gap:14px;align-items:flex-start;background:var(--card);border:1px solid var(--line);
      border-radius:13px;padding:14px 16px;margin-bottom:10px;box-shadow:0 1px 2px rgba(28,30,60,.04);animation:rise .3s ease both}
 @keyframes rise{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
 .rank{flex:0 0 26px;height:26px;border-radius:8px;background:#f1f2f8;color:#9498a8;font-size:12px;font-weight:700;
      display:grid;place-items:center;margin-top:2px}
 .hbody{flex:1;min-width:0}
 .ly{font-size:16.5px;color:var(--ink);margin-bottom:2px}
 .mt{font-size:12.5px;color:var(--faint)}
 .why{font-size:12px;color:var(--faint);margin-top:7px;padding-top:7px;border-top:1px dashed var(--line)}
 .score{flex:0 0 84px;text-align:right}
 .scnum{font-size:14px;font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums}
 .scbar{height:5px;background:#edeef4;border-radius:3px;margin-top:6px;overflow:hidden}
 .scbar i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .45s ease}
 .empty{color:var(--faint);font-size:14px;padding:8px 2px}

 /* think panel */
 .think{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 4px 16px rgba(28,30,60,.05)}
 .think-head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:16px}
 .think-kicker{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
 .think-mode{font-size:13px;font-weight:700;color:var(--accent)}
 .flow{display:flex;flex-direction:column}
 .fstep{display:flex;gap:13px;position:relative;padding-bottom:18px}
 .fstep:last-child{padding-bottom:0}
 .fstep:not(:last-child)::before{content:'';position:absolute;left:13px;top:28px;bottom:2px;width:2px;background:var(--line)}
 .fnum{flex:0 0 27px;height:27px;border-radius:50%;background:var(--accent);color:#fff;display:grid;place-items:center;
      font-size:12px;font-weight:700;z-index:1}
 .ftitle{font-size:13.5px;font-weight:600;margin-bottom:4px}
 .fcontent{font-size:13px;color:var(--muted)}
 .pill{display:inline-block;background:#f6f7fb;border:1px solid #e3e5ee;border-radius:6px;padding:1px 8px;margin:2px 3px 2px 0;font-size:12.5px}
 .pill.hit{border-color:#bfe6c9;color:var(--ok);background:#eef9f1}
 .pill.miss{opacity:.5;text-decoration:line-through}
 mark{background:#fff0bd;color:#7a5b00;border-radius:3px;padding:0 2px}

 /* learn */
 .learn{padding:18px 0 90px}
 .learn h2{font-size:22px;letter-spacing:-.02em;margin:36px 0 6px}
 .learn .lead{color:var(--muted);margin:0 0 16px;max-width:680px}
 .methods{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;box-shadow:0 1px 2px rgba(28,30,60,.03)}
 .tag{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
      color:var(--accent);background:var(--accentSoft);padding:3px 9px;border-radius:6px;margin-bottom:10px}
 .card h3{margin:0 0 8px;font-size:16px}
 .card p{margin:0 0 8px;font-size:14px;color:#3d4250}
 .card .tradeoff{font-size:13px;color:var(--faint);margin:0}
 code{background:#f1f2f7;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:13px;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
 .formula{text-align:center;font-size:18px;padding:14px;background:#fafbff;border:1px solid var(--line);border-radius:10px;margin:10px 0}
 table{width:100%;border-collapse:collapse;font-size:14px;margin:8px 0;background:var(--card);border-radius:10px;overflow:hidden}
 th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left}
 th{color:var(--muted);font-weight:600;background:#fafbfd;font-size:13px}
 tr:last-child td{border-bottom:0}
 .note{color:var(--faint);font-size:13px;margin-top:10px}

 @media(max-width:860px){
   h1{font-size:30px}
   .layout{grid-template-columns:1fr;gap:20px}
   .think-col{position:static}
 }
</style></head><body>

<header class="hero"><div class="wrap">
  <div class="kicker">Applied NLP · embeddings demo</div>
  <h1>Lyric Search</h1>
  <p class="tagline">Find a song line by what it <b>means</b>, what it <b>says</b>, or how it's <b>spelled</b> —
     searched across __N__ lines of real lyrics.</p>
  <div class="searchbar">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round"><circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.3-4.3"></path></svg>
    <input id="q" placeholder="Try: feeling heartbroken and alone" autofocus>
    <button onclick="run()">Search</button>
  </div>
  <div class="segmented" id="modes">
    <button class="seg on" data-m="glove">Meaning<small>synonyms &amp; mood</small></button>
    <button class="seg" data-m="tfidf">Keywords<small>exact words</small></button>
    <button class="seg" data-m="tri">Fuzzy<small>typo-tolerant</small></button>
  </div>
  <div class="chips" id="chips"></div>
</div></header>

<main class="wrap layout">
  <section>
    <p class="col-head">Results <span id="rescount"></span></p>
    <div id="out"></div>
  </section>
  <aside class="think-col">
    <div class="think" id="think"></div>
  </aside>
</main>

<section class="wrap learn">
  <h2>How it works</h2>
  <p class="lead">A computer can't compare text by meaning directly. So we turn every line into a
     <b>vector</b> (a list of numbers), turn your query into a vector the same way, and rank lines by
     <b>cosine similarity</b> — the angle between the two. All three modes share that idea; they differ only
     in how the vector is built.</p>

  <h2>The three methods</h2>
  <div class="methods">
    <div class="card"><span class="tag">Meaning</span>
      <h3>Averaged GloVe embeddings</h3>
      <p>GloVe gives every word 100 numbers, learned so words used in similar contexts sit close together —
         <code>sad</code>, <code>lonely</code> and <code>heartbroken</code> end up near each other.</p>
      <p>A line's vector is the <b>idf-weighted average</b> of its word vectors — rare content words count more,
         fillers like <code>the</code>/<code>and</code> barely count. That's why it finds synonyms with no shared words.</p>
      <p class="tradeoff">Trade-off: ignores word order; can't embed misspelled words.</p></div>
    <div class="card"><span class="tag">Keywords</span>
      <h3>tf-idf bag-of-words</h3>
      <p>Each line is weighted by <b>tf-idf</b> = how often a word appears here × how <i>rare</i> it is overall.
         Rare, specific words (a name, a place) count more; stop-words are dropped.</p>
      <p>Results come from cosine between the query's tf-idf vector and each line's.</p>
      <p class="tradeoff">Trade-off: blind to synonyms and to typos.</p></div>
    <div class="card"><span class="tag">Fuzzy</span>
      <h3>Character trigrams</h3>
      <p>Each word is broken into 3-letter chunks: <code>london → lon · ond · ndo · don</code>.
         Two strings are similar if they share many chunks.</p>
      <p>A typo only changes a few chunks, so <code>heartbrokn</code> still matches <code>heartbroken</code>.</p>
      <p class="tradeoff">Trade-off: matches spelling, not meaning.</p></div>
  </div>

  <h2>How one score is computed</h2>
  <div class="card">
    <p>Every mode ends the same way — <b>cosine similarity</b> between two vectors:</p>
    <div class="formula"><code>cos(a, b) = (a · b) / (‖a‖ · ‖b‖)</code></div>
    <p style="margin-bottom:0">It ranges 0 → 1 here (higher = closer). We sort all lines by it and show the top 8.
       The number and bar on each result <i>are</i> this score. Switch modes on the same query to compare.</p>
  </div>

  <h2>The data pipeline</h2>
  <table>
    <tr><th>Step</th><th>What happens</th><th>File</th></tr>
    <tr><td>1 · Load</td><td>57,650 songs (artist, title, lyrics)</td><td>spotify_millsongdata.csv</td></tr>
    <tr><td>2 · Split</td><td>lyrics → lines, clean, dedup → 180k lines</td><td>preprocess.py</td></tr>
    <tr><td>3 · Index</td><td>tf-idf + averaged GloVe per line</td><td>build_index.py</td></tr>
    <tr><td>4 · Index</td><td>character-trigram tf-idf per line</td><td>build_trigram.py</td></tr>
    <tr><td>5 · Ship</td><td>bake 5k lines + vectors into this page</td><td>export_web.py</td></tr>
  </table>
  <p class="note">We search the <b>lyric text</b> (one line per document). Artist and title are only source labels —
     not searched. Techniques are Day-2 only: tokenization, tf-idf, pretrained embeddings, cosine, character n-grams.</p>
</section>

<script>
const GLOVE=__GLOVE__, META=__META__, VECS=__VECS__, IDF=__IDF__, TOKS=__TOKS__,
      WEIGHT=__WEIGHT__, WDEFAULT=__WDEFAULT__, DIM=100;
const MODE_LABEL={glove:"Meaning",tfidf:"Keywords",tri:"Fuzzy"};
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
function embed(text){const ts=toks(text);const v=new Float32Array(DIM);let den=0; // idf-weighted average
  for(const t of ts){const g=GLOVE[t];if(g){const wt=WEIGHT[t]||WDEFAULT;for(let i=0;i<DIM;i++)v[i]+=wt*g[i];den+=wt}}
  if(!den)return null;let s=0;for(let i=0;i<DIM;i++){v[i]/=den;s+=v[i]*v[i]}
  s=Math.sqrt(s)||1;for(let i=0;i<DIM;i++)v[i]/=s;return v;}
function nearestWords(qv,k){const out=[];
  for(const w of GWORDS){const g=GLOVE[w];let d=0;for(let i=0;i<DIM;i++)d+=g[i]*qv[i];out.push([d/GNORM[w],w])}
  out.sort((a,b)=>b[0]-a[0]);return out.slice(0,k).map(x=>x[1]);}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function hl(line,words){let h=esc(line);for(const w of words){if(!w)continue;
  h=h.replace(new RegExp('\\b('+w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')\\b','ig'),'<mark>$1</mark>')}return h;}
function pills(arr,cls){return arr.map(x=>`<span class="pill ${cls||''}">${esc(x)}</span>`).join('');}
function sp(arr){return arr.map(x=>x.replace(/ /g,'␣'));}  // show spaces in chunks

function renderThink(steps){
  const head=`<div class="think-head"><span class="think-kicker">How this search works</span>
    <span class="think-mode">${MODE_LABEL[MODE]}</span></div>`;
  const body=steps.map(s=>`<div class="fstep"><span class="fnum">${s.n}</span><div>
    <div class="ftitle">${s.t}</div>${s.html?`<div class="fcontent">${s.html}</div>`:''}</div></div>`).join('');
  document.getElementById('think').innerHTML=head+`<div class="flow">${body}</div>`;
}

function run(){
  const q=document.getElementById('q').value.trim();
  const out=document.getElementById('out');const cnt=document.getElementById('rescount');
  if(!q){renderThink([{n:'·',t:'Waiting for a query',html:'Type something above to watch the pipeline.'}]);
    out.innerHTML='<div class="empty">Your results will appear here.</div>';cnt.textContent='';return;}
  const qt=toks(q);
  let scored=null, detail=null, steps=[];

  if(MODE==='glove'){
    const found=qt.filter(t=>GLOVE[t]), miss=qt.filter(t=>!GLOVE[t]);
    const qv=embed(q);
    steps=[{n:1,t:'Tokenize the query',html:pills(qt)},
      {n:2,t:'Look up GloVe vectors',html:`found <b>${found.length}/${qt.length}</b> ${miss.length?'· skipped '+pills(miss,'miss'):''}`},
      {n:3,t:'Average into one meaning vector',html:`${found.length} word vectors → a single 100-d vector`}];
    if(qv){steps.push({n:4,t:'The query reads as',html:pills(nearestWords(qv,6),'hit')+' <span style="color:var(--faint)">(nearest words)</span>'},
      {n:5,t:'Rank by cosine',html:`compare to all ${META.length} lines, keep the closest`});
      scored=VECS.map((vec,i)=>{let d=0;for(let k=0;k<DIM;k++)d+=vec[k]*qv[k];return[d,i]});}
    else steps.push({n:4,t:'No known words to embed',html:'Try the Fuzzy mode for typos.'});
  }
  else if(MODE==='tfidf'){
    const kept=qt.filter(t=>IDF[t]), dropped=qt.filter(t=>!IDF[t]);
    steps=[{n:1,t:'Tokenize the query',html:pills(qt)},
      {n:2,t:'Keep in-vocabulary words',html:pills(kept,'hit')+(dropped.length?' · dropped '+pills(dropped,'miss'):'')},
      {n:3,t:'Weight by idf (rarer = heavier)',html:kept.map(w=>`${esc(w)} <b>${IDF[w]}</b>`).join(' · ')||'—'},
      {n:4,t:'Rank by tf-idf cosine',html:'matched words are <mark>highlighted</mark> in results'}];
    if(kept.length){let qn=0;const qw={};for(const w of kept){qw[w]=IDF[w];qn+=IDF[w]*IDF[w]}qn=Math.sqrt(qn)||1;
      scored=TOKS.map((ts,i)=>{let d=0;const seen=new Set();
        for(const w of ts){if(qw[w]&&!seen.has(w)){seen.add(w);d+=qw[w]*IDF[w]}}return[d/(lineNorm[i]*qn),i]});}
    detail={type:'words',words:kept};
  }
  else{
    const qs=trigrams(q);
    steps=[{n:1,t:'Split each word into 3-letter chunks',html:pills(sp([...qs]).slice(0,16))+(qs.size>16?' …':'')},
      {n:2,t:'Do the same for every line',html:'per word, within word boundaries'},
      {n:3,t:'Score by shared chunks (Dice)',html:'typos still share most chunks, so they match'}];
    scored=lineTri.map((ls,i)=>{let inter=0;for(const g of qs)if(ls.has(g))inter++;return[2*inter/(qs.size+ls.size||1),i]});
    detail={type:'tri',qs:qs};
  }

  renderThink(steps);
  if(!scored){out.innerHTML='<div class="empty">Nothing to rank in this mode.</div>';cnt.textContent='';return;}
  scored.sort((a,b)=>b[0]-a[0]);
  // diversify: drop duplicate lines and cap each song to 2 results
  const top=[],seenL=new Set(),perSong={};
  for(const pair of scored){if(pair[0]<=0)break;const m=META[pair[1]];const k=m.l.trim().toLowerCase();
    if(seenL.has(k)||(perSong[m.s]||0)>=2)continue;
    seenL.add(k);perSong[m.s]=(perSong[m.s]||0)+1;top.push(pair);if(top.length>=8)break;}
  cnt.textContent=top.length?`· top ${top.length}`:'';
  if(!top.length){out.innerHTML='<div class="empty">No matches in this mode — try another mode or other words.</div>';return;}
  out.innerHTML=top.map(([sc,i],idx)=>{const m=META[i];
    let line=esc(m.l), why='';
    if(detail&&detail.type==='words'){line=hl(m.l,detail.words);
      const matched=detail.words.filter(w=>new RegExp('\\b'+w+'\\b','i').test(m.l));
      why=matched.length?`matched: ${matched.join(', ')}`:'';}
    else if(detail&&detail.type==='tri'){const ls=lineTri[i];const sh=[...detail.qs].filter(g=>ls.has(g));
      why=`shared chunks: ${sp(sh).slice(0,8).join(' · ')}${sh.length>8?' …':''}`;}
    const pct=Math.round(Math.max(0,Math.min(1,sc))*100);
    return `<div class="hit" style="animation-delay:${idx*30}ms">
      <div class="rank">${idx+1}</div>
      <div class="hbody"><div class="ly">"${line}"</div><div class="mt">${esc(m.a)} · ${esc(m.s)}</div>
        ${why?`<div class="why">${esc(why)}</div>`:''}</div>
      <div class="score"><div class="scnum">${sc.toFixed(2)}</div><div class="scbar"><i style="width:${pct}%"></i></div></div>
    </div>`}).join('');
}

document.querySelectorAll('.seg').forEach(el=>el.onclick=()=>{
  document.querySelectorAll('.seg').forEach(e=>e.classList.remove('on'));
  el.classList.add('on');MODE=el.dataset.m;run();});
const examples=["feeling heartbroken and alone","heartbrokn and lonley","i want to dance all night",
                "missing my hometown","money and power"];
const chips=document.getElementById('chips');
examples.forEach(e=>{const c=document.createElement('span');c.className='chip';c.textContent=e;
  c.onclick=()=>{document.getElementById('q').value=e;run()};chips.appendChild(c)});
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')run()});
run();
</script></body></html>"""

html = (TEMPLATE.replace("__GLOVE__", json.dumps(glove))
                .replace("__META__", json.dumps(meta, ensure_ascii=False))
                .replace("__VECS__", json.dumps(linevecs))
                .replace("__IDF__", json.dumps(idf))
                .replace("__TOKS__", json.dumps(toks))
                .replace("__WEIGHT__", json.dumps(weight))
                .replace("__WDEFAULT__", json.dumps(default_w))
                .replace("__N__", f"{len(meta):,}"))
with open("artifacts/search.html", "w") as f:
    f.write(html)
print("saved -> artifacts/search.html (production UI)")
