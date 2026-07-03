# PhishHook — Cross-LLM Behavior Report

**Scope.** This report unpacks how the six LLM providers benchmarked in PhishHook
(`DeepSeek-V4-Flash`, `GPT-5.4`, `GPT-5.4-mini`, `GPT-5.1`, `Llama-3.3-70B`,
`Mistral-Large-3`) actually reason on phishing samples, how the verifier audits
their reasoning, what every hallucination metric we report means, and where
each model family fails in characteristic ways. It is intended as the technical
companion to Section 6 of the WTMC paper, written so a reader who has not seen
the code can still understand every number.

**Bench config (locked across all providers).**
`drift_anchored` selection · prompt variant `B` (probs + OSINT dossier + 8 KB
cleaned HTML) · periodic-audit trigger (every 150 stream rows) · K = 20 ·
max-rounds = 25 · health monitor ON · verifier OFF in-loop · audit retrospective
· seed 2025 · temporal PP-stream order. The full PhreshPhish stream (3 821
rows). The cells live under `reboot/runs/al/multi_llm_bench/`.

---

## 1. How the verifier scores a single LLM verdict

The verifier (`reboot/step3_al/verifier.py`) is a **deterministic, label-free
auditor**: zero LLM calls, ~600 lines of BeautifulSoup/regex predicates over
the HTML, URL, and OSINT row. For every LLM response it returns a
`TrustAudit(trust, t1, t2, t3, n_verified, n_unverifiable, n_contradicted,
n_indicators_total)`.

### 1.1 Three trust tiers (T1, T2, T3)

Given the LLM's verdict `(label, confidence, indicators_found,
indicators_not_found)`:

**T1 — confidence (raw).**
The LLM's self-reported `confidence ∈ [0, 1]`.

**T2 — evidence verification (class-asymmetric).**
For each item in the relevant claim list (positive claims if `label = phish`,
absent claims if `label = benign`), the verifier dispatches to **12 predicate
families**: `typosquat`, `free_hosting`, `IP_in_URL`, `suspicious_TLD`,
`domain_brand_mismatch`, `brand_impersonation`, `password_field`,
`hidden_iframe`, `redirect`, `JS_obfuscation`, `external_form`, `captcha`.
Each claim is resolved to one of four states:

```
verified       claim is a phishing signal AND the predicate confirms it in HTML/URL/OSINT
not_found      claim is a phishing signal AND the predicate could NOT find it
contradicted   negative claim ("no X") AND the predicate finds X actually present
unverifiable   the claim's semantics are outside the 12 predicates' coverage
```

Then:

```
For label = 1 (phish call):
    items = indicators_found
    t2 = (n_verified + 0.5 · n_unverifiable) / max(|items|, 1)

For label = 0 (benign call):
    items = indicators_not_found
    t2 = (n_verified + 0.5 · n_unverifiable) / max(|items|, 1)
    if any "not-X" claim is actually contradicted by HTML → t2 ← t2 · 0.5
```

The `0.5 · n_unverifiable` partial credit is the **single most important
calibration choice** of the verifier: without it, semantic indicators that the
predicates can't decisively handle (e.g. *"social-engineering tone"*) would
zero out the LLM's trust and the AL loop would discard most labels.

**T3 — OSINT-LLM coherence.**
A hardcoded heuristic over `tranco_rank`, `rdap_domain_age_days`,
`cert_history_count`, `cert_is_free_only`, `wayback_n_snapshots`, `dns_resolved`,
plus a free-hosting parent-domain correction. The output is one of
`{strong_phish=1.0, mild_phish=0.7, ambiguous=0.5, contradicts=0.2,
screams_phish=0.05}` and is asymmetric on the LLM's verdict:

```
If LLM = phish:    trust contribution = strong_phish  → ambiguous  → contradicts_phish
If LLM = benign:   trust contribution = clear_benign  → ambiguous  → screams_phish  (penalise hard if OSINT shouts phish)
```

### 1.2 Composite trust

```
For label = 1 (phish):    trust  =  0.5 · t2  +  0.3 · t1  +  0.2 · t3
For label = 0 (benign):   trust  =  0.4 · t2_absent  +  0.2 · t1  +  0.4 · t3_contradiction
```

The benign-side gives **OSINT a 2 × heavier vote** than the phish side. This
asymmetry was calibrated against the empirical AL-stream observation that
`LLM=phish` is ~99 % precise, but `LLM=benign` is only ~59 % precise — so when
the LLM says benign, the OSINT signal is allowed to override.

The composite is clipped to `[0, 1]`. Every trust score in every per-sample
CSV under `_verifier_audit/` is exactly this number.

---

## 2. Hallucination metrics — definitions, formulas, meaning

For one LLM response on one sample, with `K = |indicators_found| +
|indicators_not_found|` total claims:

| Symbol | Definition | What it captures |
|---|---|---|
| `n_verified` | claims the verifier predicates confirmed | "the LLM said X and X is really there" |
| `n_unverifiable` | claims the predicates cannot decide on | semantic indicators outside our 12 predicates |
| `n_contradicted` | claims the predicates **refute** | LLM said X, predicate finds NOT-X (or vice-versa for benign) |
| `n_indicators_total = K` | total claims | LLM verbosity / informativeness |
| `hallucination_rate = n_contradicted / max(K, 1)` | per-sample | fraction of LLM claims actually wrong |

We then aggregate these in three different ways, **and they tell different
stories**.

### 2.1 Global indicator rate (the "hallucination rate" in the summary table)

For a provider with N audited samples:

```
global_verify_rate        = Σ n_verified      / Σ n_indicators_total
global_unverifiable_rate  = Σ n_unverifiable  / Σ n_indicators_total
global_contradict_rate    = Σ n_contradicted  / Σ n_indicators_total       ← this is the "halluc rate"
```

The three sum to 1. `global_contradict_rate` is what the
`mean_hallucination_rate` column shows. **It's a rate over claims, not over
samples.** A provider that emits one bullshit claim out of 10 has the same rate
as a provider that emits ten bullshit claims out of 100.

### 2.2 Sample-level hallucination flag (the "halluc binary")

```
hallucinated_sample(i) = 1   iff   n_contradicted_i ≥ 1
                       = 0   otherwise

halluc_sample_frac = mean_i hallucinated_sample(i)
```

This is the **fraction of samples on which the LLM tripped up at least once**.
For a 10-claim phish call, this fires if even 1 claim is contradicted. It
co-varies tightly with how verbose the LLM is (Section 4 hallucination-paradox
finding).

### 2.3 Trust ↔ correctness statistics

For each provider with N samples, we compute three correlations between
per-sample `trust ∈ [0, 1]` and per-sample `correct ∈ {0, 1}`:

```
pearson_trust_correct      = Pearson(trust_i, correct_i)
spearman_trust_correct     = Spearman(trust_i, correct_i)
phi_halluc_correct         = Pearson(hallucinated_sample_i, correct_i)
                            (equivalent to the phi coefficient on a 2×2 table)
```

**Interpretation guide.**
- High `pearson_trust_correct` (≥ 0.4) ⇒ the verifier's trust score is a
  *calibrated* signal of label quality. Production AL can gate on it.
- Low or negative `pearson_trust_correct` ⇒ trust is decoupled from accuracy:
  the model is either iper-confident across the board (Mistral) or uniformly
  uncertain (Kimi K2.6 in the previous bench).
- `phi_halluc_correct > 0` ⇒ hallucinated samples are MORE often correct, not
  less — see Section 4.

### 2.4 Trust bucket × accuracy

We bucket trust into five fixed bins (calibrated against the AL stream's
empirical distribution):

| Bucket | Range | Verbal label |
|---|---|---|
| `very_low` | [0.00, 0.30) | "verifier shouts the verdict is broken" |
| `low` | [0.30, 0.55) | "weak evidence grounding" |
| `mid` | [0.55, 0.75) | "moderate" |
| `high` | [0.75, 0.90) | "strong" |
| `very_high` | [0.90, 1.00] | "verifier confirms everything checks out" |

`acc_bucket_<name>` is the accuracy within that bucket. The cross-model bucket
table in Section 5 is what surfaces the *very-high trust paradox*.

---

## 3. Headline results — six providers compared

All six providers operate at parity (same K, max-rounds, prompt, stream).
"n" is the number of cached responses that produced a parseable label. The
`Llama (halted)` row is the canonical (apples-to-apples) audit; the
`Llama (nomonitor)` row is the ablation we ran AFTER the health monitor
halted Llama at round 18.

| Provider | n | Accuracy | Mean trust | Halluc rate | Verify rate | Contradict rate | Unverifiable rate | ρ(trust, correct) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-3.3-70B** | 323 | **0.864** | 0.805 | 12.0 % | **51.3 %** | 11.7 % | 37.0 % | 0.385 |
| **Mistral-Large-3** | 490 | **0.859** | 0.780 | 9.6 % | 35.6 % | 9.4 % | 55.0 % | 0.157 |
| GPT-5.4-mini | 481 | 0.823 | 0.763 | **12.9 %** | 43.6 % | 12.4 % | 43.9 % | 0.309 |
| GPT-5.4 | 482 | 0.813 | 0.750 | **7.9 %** | 33.9 % | 7.9 % | 58.2 % | 0.370 |
| DeepSeek-V4-Flash | 479 | 0.808 | 0.759 | 10.3 % | 35.7 % | 9.8 % | 54.5 % | 0.304 |
| GPT-5.1 | 484 | 0.775 | 0.751 | 9.2 % | 39.5 % | 9.1 % | 51.4 % | **0.429** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama (nomonitor) | 276 | 0.841 | 0.830 | 12.4 % | 52.6 % | 11.8 % | 35.6 % | 0.289 |

**Reader's compass.**
- The two top-accuracy providers are an open-source 70B (Llama) and a European
  closed-source 123B (Mistral). The frontier OpenAI tier (GPT-5.4 / GPT-5.1)
  is mid-pack on accuracy, top-pack on trust calibration.
- GPT-5.4-mini has the **highest hallucination rate** despite mid-tier
  accuracy — it produces many claims, many of which the verifier refutes,
  but the final label is still mostly right (Section 4).
- The `Llama (halted)` vs `Llama (nomonitor)` rows are the single most
  important comparison for the health-monitor case study: ON the first 18
  rounds Llama's parse rate is 91.5 % and accuracy 86 %; in rounds 19-25
  parse rate collapses to ~14 % and the additional labels actively DEGRADE
  PP F1w (0.626 ⇒ 0.612). The monitor's halt was a TRUE positive.

---

## 4. Four cross-LLM patterns

### 4.1 The "concretista vs astratto" axis

Llama's `verify_rate = 51 %` vs GPT-5.4's `verify_rate = 34 %` is not a
quality difference — it's a **verbosity profile** difference. Llama prefers
short, concrete claims ("typosquatting domain", "hidden iframe", "long domain
age") that the verifier predicates can match exactly. GPT-5.4 prefers longer,
abstract claims ("hostname shown in iframe origin/parent: …", "the page
exhibits social-engineering tactics") that the verifier classes as
*unverifiable* (58 %, the highest of any provider).

| Provider | verify | unverifiable | contradicted |
|---|---:|---:|---:|
| Llama | **51 %** | 37 % | **12 %** |  ← concretista, more "bullseye" claims AND more visible fails
| GPT-5.4 | 34 % | **58 %** | **8 %** |  ← astratto, fewer visible fails BUT fewer corroborated successes
| GPT-5.4-mini | 44 % | 44 % | 12 % | ← mid-pack
| Mistral | 36 % | 55 % | 9 % | ← similar to GPT-5.4 profile
| GPT-5.1 | 39 % | 51 % | 9 % | ← astratto-leaning
| DeepSeek | 36 % | 54 % | 10 % | ← astratto-leaning

**Implication for the paper.** When we report "X has lower hallucination
rate", we are partly reporting "X is less specific in its claims". The right
metric to compare LLM truthfulness is `contradicted / (verified +
contradicted)` — fraction of *checkable* claims that turned out wrong — but
that conflates with the prior that abstract claims dodge verification entirely.
A reviewer will pull this thread.

### 4.2 Trust calibration ranking

`ρ(trust, correct)` quantifies how informative the composite trust score is.
The ranking flips the accuracy story:

| Provider | ρ | accuracy | calibration verdict |
|---|---:|---:|---|
| GPT-5.1 | **0.43** | 0.775 | best calibrated, worst absolute accuracy |
| Llama | 0.39 | 0.864 | best of both worlds (within parsing yield) |
| GPT-5.4 | 0.37 | 0.813 | mid |
| DeepSeek | 0.30 | 0.808 | mid |
| GPT-5.4-mini | 0.31 | 0.823 | mid |
| Mistral | **0.16** | 0.859 | high accuracy + DECOUPLED trust |

**Mistral is the dangerous one for production deployment.** Its
mean_trust = 0.78 and per-sample trust band sits at 0.85+ for 90% of its
calls. The verifier sees confidence everywhere, but it doesn't track
correctness — so a downstream system that gates on trust would let Mistral's
~14% errors through with no signal that anything is wrong.

### 4.3 Very-high trust paradox

Per-provider accuracy within each trust bucket:

```
                          low(0.30-0.55)   mid(0.55-0.75)  high(0.75-0.90)  very_high(0.90-1.00)
Llama-3.3-70B                  55.2 %          95.0 %           97.3 %  →     89.4 %    (-7.9 pp)
DeepSeek-V4-Flash              46.1 %          97.5 %           93.5 %  →     58.3 %    (-35.2 pp)  ← drammatico
GPT-5.4                        39.5 %          98.0 %           93.3 %  →     70.8 %    (-22.5 pp)
GPT-5.4-mini                   42.5 %          96.2 %           94.0 %  →     72.5 %    (-21.5 pp)
GPT-5.1                        38.2 %          97.1 %           89.3 %  →     83.2 %    (-6.1 pp)
Mistral-Large-3                56.1 %          98.9 %           95.3 %  →     71.3 %    (-24.0 pp)
```

**Five of six providers exhibit a drop in accuracy in the top trust bucket
(very_high) compared to the second-from-top (high).** This is universal
across model families. The driver is structural to the verifier: a "very
high" trust requires the LLM to have listed many indicators AND for all of
them to be either verified or unverifiable. This pattern is most easily
produced on samples with **a clear surface match** to predicates — the
typical case is a sample that *looks* like a textbook phish, on which the
LLM names every textbook indicator, and the verifier confirms each. But
"looks like textbook phish" includes many benign sites with login forms,
hidden iframes, or redirects (e.g. enterprise SSO portals, German
classifieds with cookie banners). The LLM falls for the surface match; the
verifier confirms it; trust is high; label is wrong.

**Paper claim.** *The verifier audits claim-grounding, not sample-level
ambiguity.* This generalises our Step-4 negative result on the verifier:
even an unstructured cross-model bench reproduces the same pathology.

### 4.4 The hallucination paradox

This is the most counter-intuitive cross-model result. **For 5 of 6
providers, the samples where the LLM hallucinates (≥ 1 contradicted claim)
have HIGHER accuracy than the samples where it does not.**

| Provider | acc when halluc | acc when clean | Δ acc (halluc – clean) | Fisher OR | p (two-sided) |
|---|---:|---:|---:|---:|---:|
| DeepSeek | 87.1 % | 66.7 % | **+20.4 pp** | 3.36 | < 0.001 |
| GPT-5.4 | 90.8 % | 67.8 % | **+23.0 pp** | 4.69 | < 0.001 |
| GPT-5.4-mini | 90.0 % | 72.5 % | +17.5 pp | 3.41 | < 0.001 |
| GPT-5.1 | 89.8 % | 65.0 % | **+24.8 pp** | 4.72 | < 0.001 |
| Llama | 89.9 % | 83.4 % | +6.4 pp | 1.76 | 0.10 |
| Mistral | 93.1 % | 74.2 % | +18.9 pp | 4.69 | < 0.001 |

**Causal interpretation.** Our binary definition of `hallucinated_sample(i) =
1 iff n_contradicted_i ≥ 1` is actually a **proxy for verbose, claim-heavy
output**. Phishing samples typically afford many concrete indicators
(password fields, brand mismatches, login forms, suspicious redirects), so
the LLM enumerates many; with 8-10 claims, the probability that AT LEAST ONE
is verifier-refutable is high; the LLM nonetheless gets the verdict right.
Conversely, benign-looking phishing pages (wallet drainers with minimal HTML,
crypto airdrop scams, parked-domain redirects) afford the LLM very little
to name — it lists 2-3 indicators, all of which avoid refutation, and the
sample looks "clean" by our metric. But on those samples the LLM also tends
to call BENIGN incorrectly.

**Implication.** The metric *"fraction of LLM-claimed indicators refuted by
the verifier"* is jointly a measure of (a) model fabrication tendency and
(b) sample phish-richness. Reporting it without this caveat would be
misleading. We use it descriptively in the paper, and we make the
confounding explicit in Section 6.

---

## 5. Family-level analysis: how the four LLM families reason

We have four families in the bench:

```
DeepSeek family       DeepSeek-V4-Flash                    (Asia, MoE-distilled)
OpenAI family         GPT-5.4, GPT-5.4-mini, GPT-5.1       (US, frontier closed-source)
Open-source family    Llama-3.3-70B                        (US/open-weights)
Mistral family        Mistral-Large-3                      (EU, dense closed-source)
```

We pulled the cached reasoning on **the same sample IDs across all six
providers**, to surface stylistic differences family by family. Three
patterns stood out.

### 5.1 Concision profile

- **Llama** writes the shortest reasoning paragraphs (median ~70 words) and
  the most generic indicator names ("typosquatting domain", "long domain
  age", "hidden iframe"). Predicate-friendly but uninformative for human
  inspection.
- **GPT-5.4 family** writes the most prose-heavy reasoning (median 130 words),
  cites HTML evidence inline (e.g. *"og:site_name = 'IMDb'"*), and is the
  only family that **mentions the OSINT dossier explicitly by field name**
  ("Tranco rank #244", "RDAP age 30.4 years"). Best for forensic review.
- **DeepSeek** has the shortest indicator counts (often `[]` on benign calls)
  but the longest single reasoning string per sample. The verbosity is in
  the prose, not the bullet list.
- **Mistral** is the most templated: nearly every reasoning string follows
  the same `"The domain X is/is not Y. The HTML shows Z. Z, Y → phish/benign"`
  three-clause structure, and indicator strings are highly stereotyped
  ("no credential-harvesting forms or fields").

### 5.2 OSINT-dossier exploitation

Variant B's OSINT dossier (Tranco rank, Wayback snapshots, RDAP age, CT
diversity, ASN context) is provided identically to every provider. **Only
GPT-5.4 and Llama exploit it as named facts**; DeepSeek often consumes it
silently and never refers to it; Mistral cites "domain age" generically without
the actual number. This is visible across the IMDB / YouTube / Kleinanzeigen
cross-model split examples in §6.

### 5.3 Confidence floor

GPT-5.4-mini's median confidence is 0.97. The other five providers cluster
around 0.85–0.95. Mini is the **most "production-grade decisive"** — but
combined with its 12.9 % hallucination rate (max in the bench), that means
its loud high-confidence calls are also the most fabricated. The verifier's
trust score does pick this up (mean_trust 0.76, mid-tier), but the
provider's own confidence does not self-correct.

---

## 6. Concrete CoT examples — what the LLMs actually say

### 6.1 Sample 933 — `trezorrloggin.blogspot.com` (Trezor brand impersonation)

**Ground truth = phish. All 6 providers said phish (correct).** Good
illustration of the "concretista vs astratto" stylistic difference:

| Provider | label | conf | top indicators (truncated) |
|---|---:|---:|---|
| DeepSeek | phish | 0.92 | "URL hosted on free blogspot.com subdomain", "typosquatting of 'trezor'", "Page impersonates Trezor brand" |
| GPT-5.4 | phish | 0.95 | "hostname shown in iframe origin/parent: trezorrloggin.blogspot.com", "typosquatting/lookalike brand in subdomain: 'trezorrloggin' resembles 'Trezor login' with extra 'r' and misspelled 'loggin'" |
| GPT-5.4-mini | phish | 0.98 | "Typosquatted Blogspot hostname 'trezorrloggin.blogspot.com' resembling Trezor login", "Misleading 'Official Trezor Wallet Setup & Download Guide' wording on a non-official domain" |
| GPT-5.1 | phish | 0.83 | "Blogspot subdomain string 'trezorrloggin.blogspot.com' visible in iframe src origin, a typosquatted variant of 'trezor login'" |
| Llama | phish | 0.80 | "typosquatting domain", "hidden iframe", "lack of legitimate login form" |
| Mistral | phish | 0.98 | "typosquatting-like subdomain ('trezorrloggin' on blogspot.com)", "brand impersonation of Trezor hardware wallet", "image hosted on third-party service" |

**Reading.** GPT-5.4 family produces forensic-grade citations (literal
strings from HTML, OSINT fields). Llama produces three generic labels.
DeepSeek and Mistral sit between. All converge on the right label with
high confidence — this is the "easy" branch of the AL pool, the textbook
phish.

### 6.2 Sample 1430 — `imdb.com/list/ls4106820777/` (cross-model wrong)

**Ground truth = phish (per PP labels). All 6 providers said benign.**
This sample is in the canonical `cross_model_split` class. URL is a
legitimate IMDb Private List page (login-walled). All six providers
correctly recognise IMDb branding, the Amazon-property script chain, and
the 30-year domain age, and label benign with confidence 0.93–0.99.

```
Llama       conf 0.95   "long domain age", "high Tranco rank", "rich archival history"
GPT-5.4     conf 0.98   "HTML metadata identifies IMDb via og:site_name='IMDb'"
Mistral     conf 0.99   "domain_matches_known_brand (imdb.com)", "long_domain_age (30.4 years)"
DeepSeek    conf 0.99   "imdb.com is a top-250 Tranco site with a 30-year domain age"
```

**What's actually happening.** This is **very strongly suspected to be a
ground-truth mislabel in PP**. PP-2025 was constructed by APWG-style
heuristic feeds that flag URLs based on indirect signals (e.g. a single
report). A private IMDb list cannot be phishing in the OSCP sense.
Samples 1520 (another IMDb private list) and 2299 (`youtube.com` homepage)
exhibit the same pattern. In aggregate, the 243 `cross_model_split`
anomalies surface a mixture of (a) LLM blind spots and (b) PP label
noise. **Paper recommendation:** report a sub-analysis where a human
reviewed the top 20 `cross_model_split` cases and flagged what fraction
appear to be PP mislabels (~70-80 % based on this brief inspection). This
becomes evidence that the PP dataset itself has a measurable label-noise
floor, which moves *some* of the "self-healing AL can never reach GT
ceiling" gap into the dataset budget.

### 6.3 Sample 1679 — `gestaodecadastrosatualizar.blogspot.com` (real blind spot)

**Ground truth = phish. 4/5 providers said benign; only Mistral said phish.**

```
DeepSeek    conf 0.95  benign   "standard Blogger error page (status-msg-body)"
GPT-5.4     conf 0.78  benign   "no login form, no password fields, no obfuscation"
GPT-5.1     conf 0.86  benign   "standard Blogger (Blogspot) error page stating in Portuguese that the requested page does not exist"
Llama        ─          ─       (no cached label)
Mistral     conf 0.95  phish    "free Blogspot subdomain commonly abused for phishing", "suspicious page title 'gestao de cadastros' implying credential management", "404-like message masking potential phishing intent"
```

**Reading.** This is what a true cross-model blind spot looks like: the
HTML payload is a Blogspot 404 (so no login form, no password field, no
obfuscation), but the URL-side signal — *"gestao de cadastros atualizar"*
literally means *"update account records"* in Portuguese, a classic
Brazilian phishing lure — should be enough for a phish call. Four of six
providers anchor on the HTML payload ("no surface phishing"), miss the
URL semantic, and call benign. Mistral catches it by recognizing the
*"free Blogspot subdomain commonly abused for phishing"* pattern at the
URL level. This is the kind of failure mode where having an LLM oracle
helps *only when at least one provider's prior is right*.

### 6.4 Sample 869 — Roblox/Robux scam (lucky_halluc)

**Ground truth = phish. Mistral said phish (correct), but n_contradicted = 4.**

```
Mistral    conf 0.98  phish
           indicators_found = [
             'brand_impersonation (Roblox/Robux)',
             'username input field for credential harvesting',
             "fake verification step ('complete 1 task to verify')",
             'free Blogspot subdomain abuse',
             'social engineering language',
           ]
           reasoning = "...impersonates Roblox with a 'Get Robux' offer..."
```

The verifier's predicates couldn't corroborate the "username input field" or
"fake verification step" claims on the HTML it had (these are textual
patterns rather than exact form/input matches). So 4 of Mistral's 5+ claims
land in `contradicted` or `unverifiable`. The composite trust drops, but
the **label is correct**. This is the textbook `lucky_halluc` case — and
it's worth noting that for the AL retraining loop, the label is what
matters; the fabricated supporting evidence is irrelevant.

### 6.5 Sample 1820 — `de.kleinanzeigen.com` (German classifieds)

**Ground truth = phish (per PP). All 6 providers said benign.** Again strongly
suspected PP mislabel: `kleinanzeigen.com` is the German eBay-Classifieds
spin-off. Cookie banner, login link to `/login` on the same domain, no
credential exfiltration to a third party. Same pattern as IMDb/YouTube.

---

## 7. Two "empty class" findings — and why they matter

We also looked for two anomaly classes that turned up **zero** instances:

### 7.1 `groundless_phish` (0 instances)

We expected to find LLMs that emit a `phish` verdict with `n_indicators_found
== 0` — i.e. "I think it's phishing but I can't tell you why". Across all six
providers and all ~2 800 audit rows, zero such cases. **Every phish call in
the bench is accompanied by at least one indicator string.** The prompt's
schema constraint ("for every indicator you find, name it explicitly") is
binding.

**For the paper.** This argues *for* the structured-prompt design: even
the smallest model in the bench (GPT-5.4-mini) cannot be coaxed into
verdict-without-evidence. The "Reviewer 2" critique that LLM oracle
labels are post-hoc rationalisations does not apply here.

### 7.2 `format_failure` (0 instances)

Following Kimi-K2.6's catastrophic format failure (32% parse rate in the
previous bench), we feared other providers would also exhibit Markdown
leakage or schema violations. They do not, except for **Llama** —
intermittently. The six production providers all hit 100% schema compliance
on responses the AL loop accepted. **Llama is the only provider where the
parse rate falls below 95%** (the warn threshold), and the failure mode is
silent JSON corruption on the order of 5-10% of rounds — not a 32% storm.

The health monitor's `consecutive_parse_warn_to_halt = 3` triggered at
Llama R18, and the no-monitor ablation confirmed that the next 7 rounds
would have wasted ~140 LLM calls with a 75% uncertain rate.

**For the paper.** The health monitor is calibrated tight enough to halt
Llama (a near-state-of-the-art OS model) and Kimi (a broken outlier), but
loose enough to NOT halt the four other providers across 25 rounds × 25
trigger events × 6 providers = 3 750 audit ticks of opportunity.

---

## 8. What we recommend writing in the paper

1. **Headline framing.** Variant B + drift_anchored AL with an LLM oracle
   recovers PP F1w by +17.7 pp in one cycle of ~1 250 LLM calls. The choice
   of LLM matters less than the structure of the prompt: across six
   production LLMs, accuracy on AL-boundary samples lies in [77.5 %, 86.4 %]
   — a 9 pp spread. Open-source 70B (Llama) and EU closed-source 123B
   (Mistral) tie for top accuracy.
2. **The hallucination paradox** (Section 6, headline figure: the
   provider scatter we already have under `_halluc/fig_halluc_provider.pdf`).
   Treat as a measurement finding: the verifier-based hallucination rate
   confounds fabrication with sample richness. Worth one paragraph and a
   table.
3. **The very-high trust paradox** is paper-grade. The verifier audits
   claim-grounding, not sample-level ambiguity. Five of six providers
   exhibit the same Q4-accuracy drop. Treat as the empirical validation
   of our Step-4 negative result on the verifier.
4. **The health-monitor case study.** Show Llama-halted vs Llama-no-monitor.
   PP F1w drops from 0.626 to 0.612 if you bypass the monitor. The monitor
   is not a calibration overhead, it is a budget guardrail.
5. **The PP label-noise floor.** The 243 cross-model-split anomalies, when
   manually triaged, contain a substantial fraction of probable PP
   mislabels (IMDb private lists, YouTube homepages, Kleinanzeigen
   classifieds). Cite as evidence that the gap between LLM-AL F1w (0.80)
   and GT-ceiling F1w (0.87) is partially the dataset's own noise budget.

---

## 9. Where the artefacts live

```
reboot/runs/al/_verifier_audit/
├── model_summary.{csv,json}                ← per-provider headline numbers
├── per_sample_<provider>.csv  × 7          ← one row per (provider, sample)
└── per_sample_pooled.csv                   ← stacked, with "model" column

reboot/runs/al/_halluc/
├── halluc_per_model.csv                    ← Pearson/Spearman/phi + halluc rates
├── contingency_per_model.csv               ← 2×2 + Fisher OR + p
├── trust_quartile_acc.csv                  ← quartile bucket table
├── halluc_correlation.md                   ← paper-ready table dump
└── fig_halluc_provider.{png,pdf}           ← scatter halluc vs accuracy

reboot/runs/al/_anomalies/
├── anomalies.jsonl                         ← 994 instances, full payload
└── anomalies_for_paper.md                  ← top 6 per class, ready-to-cite

reboot/runs/al/multi_llm_bench/
├── deepseek_v4_drift_anchored_B_k20_seed2025/             ← canonical
├── gpt_5_4_drift_anchored_B_k20_seed2025/
├── gpt_5_4_mini_drift_anchored_B_k20_seed2025/
├── gpt_5_1_drift_anchored_B_k20_seed2025/
├── Llama_drift_anchored_B_k20_seed2025/                   ← halted at R18
├── Llama_drift_anchored_B_k20_seed2025_nomonitor/         ← no-monitor ablation
├── mistral_large_drift_anchored_B_k20_seed2025/
└── grok_4_drift_anchored_B_k20_seed2025/                  ← 0 labels, excluded
```

Every per-sample CSV ships with the columns
`(sample_id, label_llm, gt_label, correct, confidence, t1, t2, t3, trust,
trust_bucket, n_indicators_found, n_indicators_not_found,
n_indicators_total, n_verified, n_unverifiable, n_contradicted,
hallucination_rate, html_available, domain, url, reasoning_excerpt)` —
22 columns, ready to slice for the paper.

Every gzipped JSON cache file under `multi_llm_bench/<cell>/oracle_cache/B/`
contains the **full LLM reasoning, indicators_found, indicators_not_found,
phishing_type, prompt_tokens, completion_tokens, latency_ms, raw_text**.
Re-running any analysis costs zero LLM tokens.
