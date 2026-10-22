# TrialMatch

**The problem:** About 80% of clinical trials are delayed because they can't find enough participants. At the same time, patients and caregivers who want trials face a database built for researchers: thousands of results, dense medical jargon, and eligibility rules that are hard to read.

**The AI solution:** You describe a diagnosis in your own words. TrialMatch:
1. Pulls trials that are recruiting now from ClinicalTrials.gov (free public API, no key).
2. Ranks them with an AI language model (all-MiniLM-L6-v2 sentence embeddings) by how closely each trial matches your description, not just keyword hits.
3. Runs quick eligibility checks (age, sex) and pushes likely mismatches down the list.
4. Highlights the 3 eligibility criteria most relevant to you (the same AI model compares each criterion to your story), and decodes jargon such as ECOG, RECIST or "refractory" into plain English.

### v2: patient details + explainable eligibility checks
Optional fields: **stage, diagnosis date, biomarkers, treatment status, prior treatments, activity level (ECOG), other health conditions**, plus preferences (skip Phase 1, skip placebo, nearby sites only, hide conflicts).

**Fit score** = AI similarity × 0.55 per likely conflict × 0.92 per "check with study team" + 4 pts per matching detail (max +12).
Rules read each trial's inclusion/exclusion text, e.g. "Stage IIIB–IV", "previously untreated", "within 100 days of diagnosis", "HER2-negative", "ECOG 0–1", "untreated brain metastases". Every finding cites the exact criterion that triggered it ("Why this rank"), so the ranking can be audited. The weights live in one `W` object at the top of the script, so they're easy to tune against the baseline test set.

The model runs entirely in the browser (Transformers.js), so there are no API costs and no personal health details leave the device. Only the condition and location are sent to ClinicalTrials.gov.

**Live demo:** `https://<your-username>.github.io/<repo-name>/`

---

## Homework: next steps with the AI solution

### 1. Data that would be useful
| Data | Why it helps |
|---|---|
| Trial records: conditions, interventions, eligibility, sites, contacts | Core matching (**integrated now**) |
| Patient profile: diagnosis, stage, biomarkers, prior treatments, age, location | Personalizes the ranking (**typed in by user now**) |
| Geocoded site locations / distance to patient | Rank by travel burden, a major reason patients drop out |
| Drug information (FDA labels, approval status) for each intervention | Explain what's being tested in plain language |
| Published results from earlier phases (PubMed) | Show whether a drug has promise |
| Insurance / cost coverage for trial care | Address a key access barrier |
| Structured medical records (FHIR) | Auto-fill the profile instead of typing |
| Outcome feedback (did the user contact or enroll?) | Training signal to improve the ranking over time |

### 2. How to get it
- **ClinicalTrials.gov API v2:** free, no key (`clinicaltrials.gov/api/v2/studies`).
- **openFDA Drug Label API:** free, no key for low volume (`api.fda.gov/drug/label.json`).
- **NIH RxNorm API:** free; normalizes drug names (`rxnav.nlm.nih.gov`).
- **PubMed E-utilities:** free (`eutils.ncbi.nlm.nih.gov`).
- **Geocoding:** OpenStreetMap Nominatim or the U.S. Census geocoder (both free).
- **Medical records:** SMART on FHIR patient-access APIs (patient-authorized; longer-term).
- **Feedback data:** log anonymous "contacted site" clicks (needs a privacy review).

### 3. Next data set to integrate
**openFDA drug labels.** For each intervention listed in a trial, look up the FDA label to show whether the drug is already approved for another use and to summarize its main warnings. This helps caregivers judge risk and gives more text for the AI to rank on.

### 4. Other enhancements
- Distance-based ranking and a map view.
- Use a small instruction-tuned model, or an LLM API later, to turn each criterion into a yes / no / unknown question for the user ("Have you had chemotherapy in the last 4 weeks?").
- Save and compare trials; generate a printable "questions for your doctor" sheet.
- Evaluation: build a small labeled set of patient profiles and their truly relevant trials, then compare embedding ranking against keyword ranking (precision@5). This ties into next week's topic of what we are optimizing for and model selection.

---

## Deploy on GitHub Pages
1. Create a new public repo (e.g. `trialmatch`).
2. **Add file → Upload files** → upload `index.html` and `README.md` → **Commit changes**.
3. **Settings → Pages** → Source: *Deploy from a branch* → Branch: `main`, folder `/ (root)` → **Save**.
4. After about a minute, the site is live at `https://<your-username>.github.io/trialmatch/`.

*Not medical advice. Eligibility is decided by each study team.*
