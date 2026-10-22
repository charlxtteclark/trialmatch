// TrialMatch matching engine — shared by the app (index.html) and the evaluation page (eval.html)
// so both always run identical code.

// Scoring weights (tune these, then re-run the evaluation)
export const W = { conflict: 0.55, check: 0.92, bonus: 0.04, maxBonus: 0.12 };

// ---------- AI model (loads lazily, cached by the browser after first use) ----------
let extractorPromise = null;
function getExtractor(){
  if (!extractorPromise){
    extractorPromise = (async () => {
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2');
      env.allowLocalModels = false;
      return pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
    })().catch(err => { console.warn('Model failed to load', err); extractorPromise = null; return null; });
  }
  return extractorPromise;
}
async function embed(texts){
  const ex = await getExtractor();
  if (!ex) return null;
  const out = await ex(texts, { pooling: 'mean', normalize: true });
  return out.tolist();
}
const dot = (a,b) => a.reduce((s,v,i) => s + v*b[i], 0);

// Fallback if the model can't load: simple keyword overlap
const STOP = new Set('a an and are as at be but by for from has have i in is it my of on or that the their this to was were with who will patients patient study trial'.split(' '));
const words = t => (t.toLowerCase().match(/[a-z0-9\-]{3,}/g) || []).filter(w => !STOP.has(w));
function keywordScore(q, doc){
  const qs = new Set(words(q)); if (!qs.size) return 0;
  const ds = new Set(words(doc)); let hit = 0; qs.forEach(w => ds.has(w) && hit++);
  return hit / qs.size;
}

// ---------- ClinicalTrials.gov ----------
const FIELDS = [
  'protocolSection.identificationModule','protocolSection.statusModule','protocolSection.descriptionModule',
  'protocolSection.conditionsModule','protocolSection.designModule','protocolSection.eligibilityModule',
  'protocolSection.contactsLocationsModule','protocolSection.armsInterventionsModule'
].join(',');

async function searchTrials(cond, loc){
  const p = new URLSearchParams({
    'query.cond': cond, 'filter.overallStatus': 'RECRUITING', pageSize: '50', countTotal: 'true', format: 'json'
  });
  if (loc) p.set('query.locn', loc);
  const base = 'https://clinicaltrials.gov/api/v2/studies?';
  let r = await fetch(base + p + '&fields=' + encodeURIComponent(FIELDS));
  if (r.status === 400) r = await fetch(base + p);
  if (!r.ok) throw new Error('ClinicalTrials.gov returned ' + r.status);
  return r.json();
}

const parseAge = s => { if (!s) return null; const m = /([\d.]+)\s*(year|month|week|day)/i.exec(s); if (!m) return null;
  const n = +m[1], u = m[2].toLowerCase(); return u==='year'?n : u==='month'?n/12 : u==='week'?n/52 : n/365; };
const PHASE_LABEL = p => p === 'EARLY_PHASE1' ? 'Early Phase 1' : p.replace('PHASE', 'Phase ');

function splitCriteria(text){
  const inc = [], exc = []; let cur = inc;
  text.split(/\r?\n/).forEach(raw => {
    const line = raw.trim(); if (!line) return;
    if (/^exclusion/i.test(line)) { cur = exc; return; }
    if (/^inclusion/i.test(line)) { cur = inc; return; }
    const clean = line.replace(/^([*\-•]|\d+[.)])\s*/, '');
    if (clean.length > 3) cur.push(clean);
  });
  return { inc, exc };
}

function normalize(study){
  const ps = study.protocolSection || {};
  const id = ps.identificationModule || {}, el = ps.eligibilityModule || {}, cl = ps.contactsLocationsModule || {};
  const t = {
    id: id.nctId, title: id.briefTitle || id.officialTitle || 'Untitled study',
    summary: ps.descriptionModule?.briefSummary || '',
    conditions: ps.conditionsModule?.conditions || [], keywords: ps.conditionsModule?.keywords || [],
    phasesRaw: (ps.designModule?.phases || []).filter(p => p !== 'NA'),
    interventions: (ps.armsInterventionsModule?.interventions || []).map(i => i.name).filter(Boolean),
    criteria: el.eligibilityCriteria || '', sex: el.sex || 'ALL',
    minAge: parseAge(el.minimumAge), maxAge: parseAge(el.maximumAge), minAgeTxt: el.minimumAge, maxAgeTxt: el.maximumAge,
    locations: cl.locations || [], contacts: cl.centralContacts || []
  };
  t.phases = t.phasesRaw.map(PHASE_LABEL);
  Object.assign(t, splitCriteria(t.criteria));
  return t;
}

function nearbySites(t, loc){
  const toks = words(loc || '');
  let sites = t.locations;
  if (toks.length) sites = sites.filter(s => toks.some(w => `${s.city} ${s.state} ${s.country}`.toLowerCase().includes(w)));
  return sites.slice(0, 3);
}

// ======================================================================
// ELIGIBILITY RULES
// Each rule returns signals: { kind: 'ok' | 'no' | 'check', label, quote? }
// ======================================================================
const firstMatch = (lines, re) => lines.find(l => re.test(l));
const sig = (kind, label, quote) => ({ kind, label, quote });

// --- Age & sex (structured fields) ---
function ruleAgeSex(t, P){
  const out = [];
  if (P.age !== null){
    const fits = (t.minAge === null || P.age >= t.minAge) && (t.maxAge === null || P.age <= t.maxAge);
    const range = `${t.minAgeTxt || 'any'} – ${t.maxAgeTxt || 'no max'}`;
    out.push(sig(fits ? 'ok' : 'no', (fits ? 'Age fits ' : 'Age outside ') + range));
  }
  if (P.sex && t.sex !== 'ALL') out.push(sig(t.sex === P.sex ? 'ok' : 'no', t.sex === P.sex ? 'Sex fits' : `Only ${t.sex.toLowerCase()} participants`));
  return out;
}

// --- Stage ---
const ROMAN = { '0':0, 'i':1, 'ii':2, 'iii':3, 'iv':4, '1':1, '2':2, '3':3, '4':4 };
const TOK = '(?:0|iv|iii|ii|i|[1-4])(?:[abc](?![a-z]))?\\d?(?![a-z])(?!\\s*(?:weeks?|days?|months?|years?|mg|cm|%))';
function parseStages(text){
  const out = new Set();
  const re = new RegExp(`\\bstages?\\s*((?:${TOK}(?:\\s*(?:,|or|and|-|–|to|through|/)\\s*)*)+)`, 'gi');
  let m;
  while ((m = re.exec(text))){
    const toks = [...m[1].matchAll(/(0|iv|iii|ii|i|[1-4])[abc]?\d?\s*(-|–|to|through)?/gi)];
    toks.forEach((tk, i) => {
      const v = ROMAN[tk[1].toLowerCase()]; out.add(v);
      if (tk[2] && toks[i+1]){ const w = ROMAN[toks[i+1][1].toLowerCase()]; for (let k = Math.min(v,w); k <= Math.max(v,w); k++) out.add(k); }
    });
  }
  return out;
}
const STAGE_NAME = s => s === 0 ? 'Stage 0' : 'Stage ' + ['0','I','II','III','IV'][s];
function ruleStage(t, P){
  if (P.stage === null) return [];
  const out = [], allowed = new Set(); let src = null;
  for (const line of t.inc){
    const before = allowed.size;
    parseStages(line).forEach(s => allowed.add(s));
    const l = line.toLowerCase();
    if (/non-?metastatic/.test(l)) [0,1,2,3].forEach(s => allowed.add(s));
    else if (/\bmetastatic\b/.test(l) && !/brain|cns/.test(l)) allowed.add(4);
    if (/locally advanced/.test(l)) allowed.add(3);
    if (/early[- ]stage/.test(l)) [0,1,2].forEach(s => allowed.add(s));
    if (allowed.size > before && !src) src = line;
  }
  if (allowed.size){
    const list = [...allowed].sort().map(STAGE_NAME).join(', ');
    out.push(allowed.has(P.stage) ? sig('ok', `Stage fits (trial: ${list})`, src) : sig('no', `Trial is for ${list}`, src));
  } else {
    const title = t.title.toLowerCase();
    if (P.stage <= 2 && /metastatic|advanced/.test(title)) out.push(sig('check', 'Trial title mentions advanced or metastatic disease', t.title));
    if (P.stage === 4 && /early[- ]stage|localized|resectable/.test(title) && !/unresectable/.test(title)) out.push(sig('check', 'Trial title mentions early-stage disease', t.title));
  }
  if (P.stage === 4){
    const ex = firstMatch(t.exc, /\bmetastatic disease\b|\bstage\s*(iv|4)\b|distant metasta/i);
    if (ex && !/brain|cns|leptomening/i.test(ex)) out.push(sig('no', 'Excludes metastatic disease', ex));
  }
  return out;
}

// --- Time since diagnosis ---
const toMonths = (n, u) => { u = u.toLowerCase(); return u.startsWith('day') ? n/30 : u.startsWith('week') ? n/4.35 : u.startsWith('year') ? n*12 : n; };
function ruleDiagnosis(t, P){
  if (P.monthsSince === null) return [];
  const out = [];
  const within = /within\s+(?:the\s+)?(?:past\s+|last\s+)?(\d+)\s*(day|week|month|year)s?\s+(?:of|from|after|since)\s+(?:the\s+)?(?:initial\s+|first\s+)?diagnos/i;
  const atLeast = /diagnos\w*\s+(?:for\s+)?(?:at least|more than|≥|>=|>)\s*(\d+)\s*(day|week|month|year)/i;
  const lw = firstMatch(t.inc, within);
  if (lw){
    const [, n, u] = within.exec(lw); const max = toMonths(+n, u);
    out.push(P.monthsSince <= max ? sig('ok', `Diagnosed recently enough (within ${n} ${u}s)`, lw) : sig('no', `Must be within ${n} ${u}s of diagnosis`, lw));
  }
  const la = firstMatch(t.inc, atLeast);
  if (la){
    const [, n, u] = atLeast.exec(la); const min = toMonths(+n, u);
    out.push(P.monthsSince >= min ? sig('ok', `Diagnosed long enough ago (at least ${n} ${u}s)`, la) : sig('no', `Must be diagnosed at least ${n} ${u}s ago`, la));
  }
  return out;
}

// --- Treatment status & history ---
const TX = {
  surgery:   { name: 'surgery', re: /surg|resection/i },
  chemo:     { name: 'chemotherapy', re: /chemotherap/i },
  radiation: { name: 'radiation', re: /radiation|radiotherap/i },
  immuno:    { name: 'immunotherapy', re: /immunotherap|checkpoint inhibitor|anti-?pd-?l?1|anti-?ctla|pembrolizumab|nivolumab|atezolizumab|durvalumab|ipilimumab/i },
  targeted:  { name: 'targeted therapy', re: /targeted therap|tyrosine kinase|\btkis?\b/i },
  hormone:   { name: 'hormone therapy', re: /hormon(e|al) therap|endocrine therap|tamoxifen|aromatase inhibitor|antiandrogen|androgen deprivation/i },
  transplant:{ name: 'stem cell transplant', re: /transplant/i },
  cart:      { name: 'CAR-T', re: /car[- ]?t\b|chimeric antigen/i }
};
const NAIVE = /previously untreated|treatment[- ]na[iï]ve|untreated (?:with|for)|no prior (?:systemic |anti-?cancer )?(?:therapy|treatment|chemotherapy)|not (?:have )?(?:previously )?received (?:any )?(?:prior )?(?:systemic )?(?:therapy|treatment)|newly diagnosed/i;
const PRETREATED = /relapsed|refractory|recurren|progress(?:ed|ion|ing)?\s+(?:on|after|following|despite)|at least (?:one|1|two|2) (?:prior|previous|line)|previously treated|after failure|failed (?:at least|prior|previous)/i;
const WASHOUT = /within\s+\d+\s*(?:day|week|month)|\d+\s*(?:day|week)s?\s+(?:prior|before)|washout/i;
function ruleTreatment(t, P){
  const out = [];
  const treated = P.tx.length > 0 || ['current','remission','relapsed','refractory'].includes(P.status);
  const known = P.status !== '' || P.tx.length > 0;
  if (!known) return out;
  const naive = firstMatch(t.inc, NAIVE);
  if (naive) out.push(treated ? sig('no', 'Trial is for people not yet treated', naive) : sig('ok', 'Trial is for newly diagnosed / untreated patients', naive));
  const pre = firstMatch(t.inc, PRETREATED);
  if (pre && !naive){
    if (P.status === 'new') out.push(sig('no', 'Trial requires prior treatment or relapse', pre));
    else if (['relapsed','refractory'].includes(P.status)) out.push(sig('ok', 'Trial is for relapsed / refractory disease, like yours', pre));
    else if (P.status === 'current' || P.status === 'remission') out.push(sig('check', 'Trial mentions relapse or progression after treatment', pre));
  }
  for (const [key, tx] of Object.entries(TX)){
    const had = P.tx.includes(key);
    if (had){
      const ex = t.exc.find(l => tx.re.test(l) && /prior|previous|received|history of|treated with|exposure/i.test(l));
      if (ex) out.push(WASHOUT.test(ex) ? sig('check', `Recent ${tx.name} may need a waiting period`, ex) : sig('no', `Excludes people who had ${tx.name}`, ex));
    } else if (P.tx.length){
      const inc = t.inc.find(l => tx.re.test(l) && /prior|previous|received|after|following|progress/i.test(l));
      if (inc) out.push(sig('check', `May require prior ${tx.name}`, inc));
    }
  }
  return out;
}

// --- Biomarkers ---
const POL_RE = '(positive|negative|pos\\b|neg\\b|\\+|mutat\\w*|amplif\\w*|fusion|rearrang\\w*|wild[- ]?type|low|high|express\\w*|[-–](?![a-z0-9]))';
const polarity = w => { w = (w || '').toLowerCase();
  if (/^(negative|neg|wild|-|–)/.test(w)) return 'neg';
  if (w === 'low') return 'low';
  return w ? 'pos' : null; };
const ALIASES = { HER2: 'HER2|HER-2|ERBB2', ER: 'ER|estrogen receptor', PR: 'PR|PgR|progesterone receptor', HR: 'HR|hormone receptor', PDL1: 'PD-?L1' };
function parseBiomarkers(s){
  return s.split(/[,;]|\band\b/i).map(x => x.trim()).filter(Boolean).map(raw => {
    const m = new RegExp(`^(.*?)\\s*${POL_RE}?\\s*$`, 'i').exec(raw);
    let name = (m?.[1] || raw).trim().replace(/[-–\s]+$/, ''); const pol = polarity(m?.[2]);
    const key = name.toUpperCase().replace(/[^A-Z0-9]/g, '');
    const alt = ALIASES[key] || name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const flags = key.length <= 3 && !ALIASES[key] ? '' : (ALIASES[key] && key.length <= 2 ? '' : 'i');
    return { name, pol, alt, flags };
  }).filter(b => b.name.length >= 2);
}
function ruleBiomarkers(t, P){
  const out = [];
  const label = b => b.name + (b.pol === 'pos' ? '+' : b.pol === 'neg' ? '−' : b.pol === 'low' ? '-low' : '');
  for (const b of P.bio){
    const mention = new RegExp(`(?<![A-Za-z])(?:${b.alt})(?![A-Za-z])`, b.flags);
    const withPol = new RegExp(`(?<![A-Za-z])(?:${b.alt})\\s*[-–]?\\s*(?:\\(\\s*)?${POL_RE}`, b.flags + 'g');
    const polsIn = line => [...line.matchAll(withPol)].map(m => polarity(m[1]));
    let decided = false;
    for (const line of [...t.inc, t.title]){
      const pols = polsIn(line); if (!pols.length || !b.pol) continue;
      decided = true;
      out.push(pols.includes(b.pol) ? sig('ok', `Matches your ${label(b)}`, line)
        : sig('no', `Trial requires ${b.name}${pols[0] === 'pos' ? '+' : pols[0] === 'neg' ? '−' : '-low'}`, line));
      break;
    }
    if (!decided){
      const ex = t.exc.find(l => b.pol && polsIn(l).includes(b.pol));
      if (ex){ out.push(sig('no', `Excludes ${label(b)}`, ex)); decided = true; }
    }
    if (!decided){
      const blob = [t.title, t.summary, t.conditions.join(' '), t.keywords.join(' '), t.criteria].join(' \n ');
      const line = [t.title, ...t.inc, t.summary].find(l => mention.test(l));
      if (mention.test(blob)) out.push(sig('ok', `Mentions ${b.name}`, line));
    }
  }
  return out;
}

// --- Activity level (ECOG / Karnofsky) ---
const KPS = [100, 80, 60, 40, 20];
function ruleECOG(t, P){
  if (P.ecog === null) return [];
  for (const line of t.inc){
    let max = null, m;
    if (/ecog|performance status|zubrod|\bps\b/i.test(line)){
      if ((m = /(?:ecog|performance status|zubrod|\bps\b)[^\n]{0,60}?([0-4])\s*(?:-|–|to|or)\s*([0-4])/i.exec(line))) max = +m[2];
      else if ((m = /(?:ecog|performance status|zubrod|\bps\b)[^\n]{0,40}?(≤|<=|<|less than or equal to|less than|of)\s*([0-4])\b/i.exec(line))) max = m[1] === '<' || m[1] === 'less than' ? +m[2] - 1 : +m[2];
    }
    if (max !== null) return [P.ecog <= max ? sig('ok', `Activity level fits (ECOG ≤ ${max})`, line) : sig('no', `Requires ECOG ${max} or better (yours ≈ ${P.ecog})`, line)];
    if ((m = /karnofsky[^\n]{0,40}?(?:≥|>=|>|at least|of|score of)\s*(\d{2,3})/i.exec(line))){
      const min = +m[1];
      return [KPS[P.ecog] >= min ? sig('ok', `Activity level fits (Karnofsky ≥ ${min})`, line) : sig('no', `Requires Karnofsky ≥ ${min}`, line)];
    }
  }
  return [];
}

// --- Other health conditions (mostly exclusions) ---
const CONDS = {
  brain:      { name: 'brain / spinal cord involvement', re: /brain|\bcns\b|central nervous system|leptomening/i, soft: /untreated|active|symptomatic|unstable|progressing/i },
  heart:      { name: 'heart disease', re: /cardiac|heart failure|myocardial|cardiovascular|arrhythm|\bqtc?\b|congestive/i },
  kidney:     { name: 'kidney disease', re: /renal|kidney|dialysis/i },
  liver:      { name: 'liver disease', re: /hepatic|liver|cirrhosis/i },
  hiv:        { name: 'HIV', re: /\bhiv\b|human immunodeficiency/i },
  hep:        { name: 'hepatitis', re: /hepatitis|\bhbv\b|\bhcv\b/i },
  autoimmune: { name: 'autoimmune disease', re: /autoimmune/i },
  othercancer:{ name: 'another recent cancer', re: /(?:other|another|second|prior|previous|concurrent|additional)\s+(?:active\s+|invasive\s+|primary\s+)?(?:malignan|cancer)|history of (?:another |other )?malignan/i },
  pregnant:   { name: 'pregnancy / breastfeeding', re: /pregnan|breast[- ]?feed|lactat|nursing/i, hard: true }
};
function ruleConditions(t, P){
  const out = [];
  for (const key of P.cx){
    const c = CONDS[key]; const ex = t.exc.find(l => c.re.test(l));
    if (!ex) continue;
    if (c.hard) out.push(sig('no', `Excludes ${c.name}`, ex));
    else if (key === 'brain') out.push(c.soft.test(ex) ? sig('check', 'Brain metastases allowed only if treated / stable', ex) : sig('no', 'Excludes brain metastases', ex));
    else out.push(sig('check', `Exclusion mentions ${c.name}`, ex));
  }
  return out;
}

const RULES = [ruleAgeSex, ruleStage, ruleDiagnosis, ruleTreatment, ruleBiomarkers, ruleECOG, ruleConditions];

function scoreTrial(sim, signals, w = W){
  const nNo = signals.filter(s => s.kind === 'no').length;
  const nCk = signals.filter(s => s.kind === 'check').length;
  const nOk = signals.filter(s => s.kind === 'ok').length;
  const mult = Math.pow(w.conflict, nNo) * Math.pow(w.check, nCk);
  const bonus = Math.min(nOk * w.bonus, w.maxBonus);
  return { final: Math.max(0, Math.min(1, sim * mult + bonus)), mult, bonus, nNo, nCk, nOk };
}


// ---------- Profile ----------
export const STATUS_TEXT = { new: 'newly diagnosed, not yet treated', current: 'currently in treatment', remission: 'finished treatment, in remission', relapsed: 'relapsed / recurrent after treatment', refractory: 'refractory, progressing on treatment' };

// raw: { cond, loc, story, age, sex, stage, diag:'YYYY-MM', status, tx:[], bio:'', ecog, cx:[], prefs:[] }
// Numeric fields may be numbers, numeric strings or ''.
export function buildProfile(raw, now = new Date()){
  const num = v => v === '' || v === null || v === undefined ? null : +v;
  let monthsSince = null;
  if (raw.diag){ const [y, m] = raw.diag.split('-').map(Number); monthsSince = (now.getFullYear() - y) * 12 + (now.getMonth() + 1 - m); }
  const P = {
    cond: (raw.cond || '').trim(), loc: (raw.loc || '').trim(), story: (raw.story || '').trim(),
    age: num(raw.age), sex: raw.sex || '', stage: num(raw.stage), monthsSince,
    status: raw.status || '', tx: raw.tx || [], bioRaw: (raw.bio || '').trim(),
    ecog: num(raw.ecog), cx: raw.cx || [], prefs: new Set(raw.prefs || [])
  };
  P.bio = parseBiomarkers(P.bioRaw);
  // v2: structured details also feed the AI's description so similarity reflects them
  P.text = [
    P.cond,
    P.stage !== null ? (P.stage === 4 ? 'stage IV metastatic' : STAGE_NAME(P.stage)) : '',
    P.bioRaw,
    STATUS_TEXT[P.status] || '',
    P.tx.length ? 'previously treated with ' + P.tx.map(k => TX[k].name).join(', ') : '',
    P.story
  ].filter(Boolean).join('. ');
  // v1 (original app): only the condition and free-text story were used
  P.textV1 = [P.cond, P.story].filter(Boolean).join('. ');
  return P;
}

export function hiddenReason(t, P, signals, loc){
  const isPh1Only = t.phasesRaw.length && t.phasesRaw.every(p => /PHASE1$/.test(p));
  if (P.prefs.has('noPhase1') && isPh1Only) return 'phase';
  if (P.prefs.has('noPlacebo') && /placebo/i.test(t.interventions.join(' ') + ' ' + t.summary)) return 'placebo';
  if (P.prefs.has('nearOnly') && loc && !nearbySites(t, loc).length) return 'distance';
  if (P.prefs.has('hideConflicts') && signals.some(s => s.kind === 'no')) return 'conflict';
  return null;
}

// ---------- Ranking systems (used by the evaluation) ----------
export const trialDoc = t => [t.title, t.conditions.join(', '), t.keywords.join(', '), t.interventions.join(', '), t.summary.slice(0, 700)].join('. ');

export function runRules(t, P){
  return RULES.flatMap(rule => { try { return rule(t, P); } catch (err) { console.warn(rule.name, err); return []; } });
}

// Baseline 0: ClinicalTrials.gov's own order (no AI)
export const rankApi = trials => trials.map(t => t.id);

// v1: AI similarity on condition + story; ×0.6 if age or sex doesn't fit
export function rankV1(trials, sims, P){
  return trials.map((t, i) => {
    const bad = ruleAgeSex(t, P).some(s => s.kind === 'no');
    return { id: t.id, s: sims[i] * (bad ? 0.6 : 1) };
  }).sort((a, b) => b.s - a.s).map(x => x.id);
}

// v2: AI similarity on the enriched profile + eligibility rules (fit score)
export function rankV2(trials, sims, P, w = W){
  return trials.map((t, i) => ({ id: t.id, s: scoreTrial(sims[i], runRules(t, P), w).final, signals: null }))
    .sort((a, b) => b.s - a.s).map(x => x.id);
}

export { getExtractor, embed, dot, words, keywordScore, searchTrials, parseAge, PHASE_LABEL, splitCriteria, normalize, nearbySites,
  RULES, parseStages, STAGE_NAME, TX, parseBiomarkers, scoreTrial, ruleAgeSex };
