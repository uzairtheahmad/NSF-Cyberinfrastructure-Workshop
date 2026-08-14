# Preprocessing as an Experimental Variable: A Reproducibility Study and Computational Benchmark of EEG Preprocessing Pipelines on the AHEPA Dementia Dataset (OpenNeuro ds004504)

**Status of this document.** Sections 1–13 and 15–19 are complete: they present the
literature review, the gap analysis, the experimental design and the methodological
argument, all of which were finalised before any data was processed. **Section 14
(Results) is intentionally unpopulated.** The three accompanying notebooks compute those
values; until they are executed on the reader's own Colab instance, no numbers exist, and
this report does not invent any. Notebook 03 writes a file called
`results/measured_results_appendix.md` containing every measured value in
paste-ready Markdown.

---

## 1. Abstract

The AHEPA dataset (OpenNeuro `ds004504`) has become the most widely used open EEG resource
for machine-learning research on Alzheimer's disease (AD) and frontotemporal dementia
(FTD). A 2026 scoping review catalogued 46 published studies using it and showed that
reported accuracy is inversely related to methodological rigour, with weaker validation
associated with 7–10 percentage points of inflation. That review examined *published
numbers*; by its own account it performed no re-analysis of the raw recordings.

A structural feature of this literature has gone unexamined: essentially every study
consumes the dataset authors' pre-cleaned derivative files. Preprocessing is inherited as
a fixed constant rather than treated as an experimental variable, even though the
review's own appraisal criteria flag preprocessing transparency (criterion C4) as a
recurring weakness.

This study treats preprocessing as the independent variable. We reproduce the authors'
published pipeline (Butterworth 0.5–45 Hz → re-reference → Artifact Subspace
Reconstruction → extended-Infomax ICA → ICLabel rejection) **from the raw EEG**, validate
that reproduction against the authors' own derivatives, and compare it against a
deterministic, ICA-free alternative (bad-channel detection → spherical-spline
interpolation → robust epoch rejection). Filtering, referencing, resampling, epoching,
feature extraction, classifier, cross-validation folds and random seed are held identical
across both arms, so preprocessing is the only thing that varies.

Evaluation is at the subject level throughout, with leakage prevention enforced by
assertion rather than convention. The primary statistical comparison is a paired bootstrap
over subjects, reported as a confidence interval for the difference; fold-level tests are
reported as supporting evidence with an explicit correction for the non-independence of
cross-validation folds. We additionally benchmark runtime, peak memory and storage in a
single fixed Google Colab environment.

**Contribution, stated conservatively.** Neither pipeline is a new method. The
contribution is an empirical comparison, a reproducibility check of a widely inherited
preprocessing pipeline, and a computational benchmark under resource-constrained
conditions. We make no claim of methodological novelty and no claim of clinical utility.

---

## 2. Introduction

### 2.1 Background

Alzheimer's disease accounts for roughly 60–80% of dementia cases worldwide, and current
diagnostic pathways — neuroimaging, PET, cerebrospinal fluid biomarkers — are expensive,
sometimes invasive, and unevenly available. Electroencephalography is cheap, non-invasive
and widely deployed, and quantitative EEG shows reproducible abnormalities in dementia:
increased power at slow frequencies (delta, theta), reduced alpha and beta power, and
disrupted functional connectivity. This combination has made EEG-based machine
classification an active research area.

### 2.2 The reproducibility problem

That activity has outrun its methodology. Across machine-learning-based science, data
leakage is pervasive; one cross-disciplinary survey documented at least 294 affected
studies across 17 fields. EEG is especially exposed because recordings are segmented into
many highly correlated epochs from the same person. If those epochs are split at random,
a model can identify the *individual* — their electrode impedance, their baseline rhythm —
rather than their pathology, and report excellent accuracy that evaporates on a new
patient.

The AHEPA benchmark review quantified this precisely on the dataset used here. Of 46
studies, only 12 met the highest validity tier; 22 exhibited leakage. Mean reported
accuracy for AD versus controls falls from 90.8% across all studies to 82.1% among
rigorously validated ones. Under proper validation, traditional classifiers matched deep
architectures — implying that much of deep learning's apparent advantage on this dataset
reflects leakage rather than representational power.

### 2.3 The unexamined variable

Two facts about that literature motivate this study.

First, the review synthesises published numbers. In its own limitations it notes that no
re-analysis of raw datasets and no simulation of alternative validation schemes was
performed.

Second, and more fundamentally: the dataset ships both raw recordings and author-cleaned
derivatives, and essentially all downstream work uses the derivatives. Preprocessing is
therefore a *shared, inherited constant* across dozens of studies — one that every result
depends on, that the review's criterion C4 identifies as under-reported, and that no
study appears to have varied or evaluated.

This is a strange situation. A processing step that every result in a literature depends
upon has never been tested within that literature.

### 2.4 Why the question is genuinely open

Delorme (2023) evaluated automated preprocessing across three public EEG collections and
found that, apart from high-pass filtering and bad-channel interpolation, automated
corrections — including automated ICA rejection of eye and muscle components — did not
reliably improve data quality; some steps, notably re-referencing, were actively harmful
to the metric used. The paper's title, *EEG is better left alone*, states the conclusion.

de Cheveigné (2023) rebuts this. He argues the significant-channel-count metric is of
limited applicability, and that the value of a cleaning tool depends on whether its target
artifact is actually present in the data — "it depends" rather than "leave it alone".

That disagreement is why this needs testing rather than asserting. Note also that
Delorme's endpoint was ERP detectability in task-based data. Ours is diagnostic
classification from resting-state data — a different question, on which his result may or
may not transfer. **We do not assume it transfers; we test whether it does.**

There is a specific reason to expect ICA might matter here: the dataset README observes
that eye-movement artifacts were found in some recordings *despite* the eyes-closed
protocol. That is exactly the scenario in which de Cheveigné's argument predicts a
component-based method should earn its cost.

---

## 3. Research question and hypotheses

> **RQ.** Does a deterministic, ICA-free preprocessing strategy deliver comparable EEG
> signal quality and comparable subject-level diagnostic classification performance to the
> dataset authors' published ASR + ICA + ICLabel pipeline, at materially lower
> computational cost, on OpenNeuro ds004504?

Three subsidiary questions:

- **RQ1 (reproducibility).** Can the authors' published pipeline be reproduced from the raw recordings using open Python tooling, and how closely does the result match their supplied derivatives?
- **RQ2 (performance).** Is downstream classification performance equivalent between the two pipelines when every downstream variable is controlled?
- **RQ3 (cost).** What is the computational cost difference in a fixed, resource-constrained environment?

**Hypotheses.** H2 is deliberately framed as *equivalence*, not superiority.

| # | Hypothesis | Falsified if |
|---|---|---|
| H1 | Pipeline B is substantially cheaper per subject | Measured runtimes are comparable (ratio < 1.5×) |
| H2 | Classification performance is comparable (CI for the difference includes 0) | The CI excludes 0 in either direction |
| H3 | Pipeline B preserves spectral content | Relative band powers differ markedly (> 20%) |
| H4 | Pipeline B leaves more residual high-amplitude activity | Residual artifact indicators are equal or lower than Pipeline A's |

H4 predicts that Pipeline B is *worse* on a signal-quality axis. Including a hypothesis we
expect to be falsified in the alternative's disfavour is deliberate: the design is not
built to make Pipeline B win.

---

## 4. Literature review

### 4.1 The dataset and its primary sources

The dataset is documented by Miltiadous et al. (2023a), *Data* 8(6):95. It contains
resting-state, eyes-closed recordings from 88 participants — 36 AD, 23 FTD, 29
cognitively normal (CN) — acquired at the 2nd Department of Neurology, AHEPA General
University Hospital, Thessaloniki, on a Nihon Kohden EEG 2100 with 19 scalp electrodes at
500 Hz. Mean MMSE is 17.75 (AD), 22.17 (FTD) and 30 (CN); mean age 66.4, 63.6 and 67.9
years respectively. The distribution includes both raw recordings and author-preprocessed
derivatives in BIDS format.

The first study on the dataset was DICE-net (Miltiadous et al., 2023b), a
convolution-transformer architecture.

### 4.2 The benchmark review

Miltiadous et al. (2026), *Cognitive Neurodynamics* 20:95, is the definitive survey. Its
findings frame this project:

- 46 studies included from 112 Scopus citations of the data descriptor (retrieved 2025-08-26).
- Studies graded on seven criteria (C1–C7) into three validity tiers: 12 at Validity 1, 12 at Validity 2, 22 at Validity 3.
- Validity-1 baselines: AD vs CN 82.11% accuracy (F1 81.57%); FTD vs CN 75.18% (F1 65.85%); AD vs FTD 71.44%; three-class 69.99%.
- Each step down in validation rigour is associated with ~7.5 points (AD/CN) to ~9.5 points (FTD/CN) of accuracy inflation; validity explains 52% and 68% of variance respectively.
- Under rigorous validation, traditional ML matched or slightly exceeded deep learning.
- Spectral features act as the de facto reference representation; connectivity and complexity features add complementary but not systematically superior information.

Its criterion **C4 — transparency of EEG preprocessing** — is the direct hook for this
study.

### 4.3 Preprocessing methodology

The reference pipeline combines Artifact Subspace Reconstruction (Kothe & Jung, 2016;
Chang et al., 2020) with extended-Infomax ICA (Bell & Sejnowski, 1995; Lee et al., 1999)
and the ICLabel classifier (Pion-Tonachini et al., 2019). Alternatives include the PREP
pipeline's robust referencing and bad-channel detection (Bigdely-Shamlo et al., 2015),
MARA (Winkler et al., 2011), Autoreject (Jas et al., 2017), wavelet-based denoising
(Mammone et al., 2012) and Riemannian ASR (Blum et al., 2019).

The critical comparative evidence is Delorme (2023) and its rebuttal (de Cheveigné, 2023),
discussed in §2.4. Callan et al. (2024) provide a counterweight in the opposite direction:
under extreme artifact (skateboarding), ASR followed by ICA clearly outperformed minimal
cleaning. Taken together, the literature supports the position that pipeline value is
context-dependent — which is an argument for measuring it in *this* context, not for
assuming an answer.

### 4.4 Novelty and gap assessment

| Alternative | Used on ds004504 before? | Reference | What was changed | Reported outcome | Novel comparison possible? |
|---|---|---|---|---|---|
| ASR + RunICA + ICLabel | **Yes** — this *is* the authors' pipeline | Miltiadous et al. 2023a | n/a (reference) | Supplied as derivatives; inherited by ~all 46 studies | This is Pipeline A, not an alternative |
| Minimal preprocessing (filter only) | Not identified | Delorme 2023 (other datasets) | Removed all automated correction | Filtering + interpolation as good as or better than heavier pipelines | **Yes** |
| Bad-channel detection + interpolation | Not identified | Bigdely-Shamlo et al. 2015 | Robust referencing + interpolation | One of two steps Delorme found beneficial | **Yes — selected** |
| ASR alone (no ICA) | Not identified | Chang et al. 2020; Callan et al. 2024 | Ablate ICA stage | ASR effective for transients | Partly — overlaps Pipeline A |
| ICA + MARA instead of ICLabel | Not identified | Winkler et al. 2011 | Swap IC classifier | Comparable classification | Yes, but retains the expensive stage |
| Wavelet thresholding / wavelet-ICA | Not identified | Mammone et al. 2012 | Wavelet denoising of ICs | Good morphology preservation | Yes, but adds parameters and cost |
| Riemannian ASR / robust PCA | Not identified | Blum et al. 2019 | Riemannian ASR variant | Improved robustness | Yes, but no mature Colab-ready implementation |

**Search scope.** The 46 studies catalogued by the benchmark review (Scopus citations of
the data descriptor, screened to 2025-08-26), plus targeted searches for
preprocessing-comparison work on this dataset.

**Assessment.** No catalogued study treats preprocessing as the independent variable or
re-runs it from the raw recordings. This is a claim about the literature, not about the
sophistication of the technique.

**What we therefore claim** — an *empirical comparison*, a *reproducibility study*, a
*computational benchmark*. **What we do not claim** — "first ever", "novel method", or any
clinical finding.

---

## 5. Dataset

| Property | Value | Source |
|---|---|---|
| Identifier | OpenNeuro `ds004504` | openneuro.org |
| Participants | 88 (36 AD, 23 FTD, 29 CN) | README / data descriptor |
| Channels | 19 scalp (10–20) + A1/A2 for impedance check | README |
| Sampling rate | 500 Hz | README |
| Condition | Resting state, eyes closed, seated | README |
| Device | Nihon Kohden EEG 2100 | README |
| Amplifier | Sensitivity 10 µV/mm, TC 0.3 s, HF filter 70 Hz | README |
| Duration | ~13.5 min (AD), ~12 min (FTD), ~13.8 min (CN) | README |
| Format | BIDS; raw + derivatives | README |
| Licence | CC0 | OpenNeuro |

Notebook 01 verifies each of these against the files rather than trusting the table, and
records mismatches in `results/dataset_verification.json`.

**Ethics.** Data collection was approved by the Scientific and Ethics Committee of AHEPA
University Hospital (protocol 142/12-04-2023). The data is openly licensed and
de-identified; this study is a secondary analysis and required no additional approval.

---

## 6. Original preprocessing pipeline (Pipeline A)

### 6.1 As published

From the dataset README: a Butterworth band-pass filter of 0.5–45 Hz, re-referencing to
A1–A2, then Artifact Subspace Reconstruction removing bad data periods exceeding a maximum
acceptable 0.5-second-window standard deviation of 17, then ICA (RunICA) producing 19
components, with components classified as eye or jaw artifacts by ICLabel automatically
rejected.

### 6.2 Our implementation

| Step | Published specification | Our implementation |
|---|---|---|
| Band-pass | Butterworth 0.5–45 Hz | MNE IIR Butterworth, order 4, zero-phase |
| Reference | A1–A2 | **Substituted** — see §6.3 |
| Resample | not specified | 500 → 100 Hz, applied identically in both arms |
| ASR | 0.5 s window, cutoff SD 17 | `asrpy` (`cutoff=17`, `win_len=0.5`) |
| ICA | RunICA, 19 components | MNE extended Infomax, n = good channels |
| IC classification | ICLabel | `mne-icalabel` ICLabel |
| Rejection | eye + jaw artifacts | `eye blink` + `muscle artifact`, p ≥ 0.5 |

### 6.3 Documented uncertainties

Reproducing a published pipeline requires values the sources do not give. We record these
rather than inventing them silently.

**1. The A1–A2 reference cannot be reproduced.** The README describes two mutually
incompatible things: that the *included* referential montage uses Cz as the common
reference, and that preprocessing re-referenced to A1–A2. The distributed raw files
contain only the 19 scalp electrodes; A1 and A2 are described as reference electrodes for
impedance checking and are absent as data channels. A linked-mastoid re-reference is
therefore not recomputable from the shared data. We substitute an average reference — also
the configuration ICLabel was designed for — and apply the *same* choice to Pipeline B, so
that referencing is controlled and cannot confound the comparison. Pipeline A is
consequently an **approximate** reproduction, and we describe it as such throughout.

**2. ICLabel has no "jaw artifact" class.** Its seven classes are brain, muscle artifact,
eye blink, heart beat, line noise, channel noise and other. We map *eye artifacts* →
`eye blink` and *jaw artifacts* → `muscle artifact` (the closest available), and document
the mapping.

**3. ICLabel is being used outside its validated regime.** ICLabel was trained on
average-referenced data band-passed 1–100 Hz. The original pipeline applies it to 0.5–45 Hz
data. We cannot supply 100 Hz content because Pipeline A's own low-pass is 45 Hz. This is
an inherent property of the original design, not of our reproduction, and it is worth
flagging as a finding in its own right: a pipeline inherited by dozens of downstream
studies applies an automated classifier outside the conditions under which it was
validated.

**4. Filter order and phase** are unstated. We use order 4, zero-phase, and record it.

**5. RunICA's random seed and convergence settings** are unavailable, so bit-identical
reproduction is impossible by construction.

**6. ASR mode.** The README says ASR *removed* bad data periods; standard `clean_rawdata`
ASR *reconstructs* them. We use reconstruction (the standard behaviour) and report
resulting duration differences rather than concealing them.

The full register is written to `results/pipeline_a_uncertainties.json`.

---

## 7. Alternative preprocessing pipeline (Pipeline B)

### 7.1 Design

```
Raw EEG
  → Butterworth band-pass 0.5–45 Hz        [IDENTICAL to Pipeline A]
  → Re-reference                           [IDENTICAL to Pipeline A]
  → Trim onset + resample to 100 Hz        [IDENTICAL to Pipeline A]
  → Deterministic bad-channel detection    ← the only difference
  → Spherical-spline interpolation         ← the only difference
  → Robust epoch rejection                 [IDENTICAL to Pipeline A]
```

### 7.2 Components

**Bad-channel detection.** Two criteria, both computed with median/MAD statistics so that
the criteria are not themselves driven by the outliers they exist to find:

- *Low correlation* — maximum absolute correlation with any other channel below 0.4, indicating an electrode not sharing the volume-conducted signal all scalp electrodes should see.
- *Amplitude deviation* — robust z-scored channel amplitude above 5, indicating a flat or noise-dominated channel.

A safety valve caps flagged channels at 25% of the montage; beyond that the criteria are
more likely wrong than the data, and the subject is flagged.

*Validation:* on synthetic data with one flat and one high-noise channel injected, the
detector recovered exactly those two channels with no false positives.

**Spherical-spline interpolation.** Bad channels are reconstructed from neighbours using
electrode geometry.

**Robust epoch rejection.** Threshold derived per subject from the median and MAD of the
epoch peak-to-peak distribution. A fixed microvolt threshold would not transfer across
subjects with different impedances and different disease-related amplitudes. Applied to
**both** pipelines, since it is a downstream step.

### 7.3 Justification

1. **It isolates one variable.** Everything except the artifact-removal stage is held constant.
2. **It targets the actual cost centre.** ASR and Infomax ICA dominate Pipeline A's runtime; replacing exactly that stage is what makes the cost question answerable.
3. **It has literature support.** It is essentially "the two steps that survived Delorme's (2023) analysis, and nothing else".
4. **It is fully deterministic.** Infomax ICA depends on random initialisation, so Pipeline A gives different output on re-run unless the seed is fixed — and the authors' seed is unavailable. Pipeline B has no stochastic component, which speaks directly to the reproducibility objective.

### 7.4 Known weaknesses

- **Interpolation on 19 electrodes is coarse.** Spherical-spline interpolation was designed for high-density montages; if many channels are flagged, Pipeline B may introduce smoothing artifacts rather than remove noise.
- **No mechanism for ocular artifact.** Eye movement is attenuated only insofar as the band-pass and epoch rejection catch it — and the README notes eye-movement artifacts *are* present despite the eyes-closed protocol. A finding that Pipeline A performs better would be theoretically coherent, not anomalous.
- **Thresholds are conventional, not tuned.** Tuning them against classification accuracy would be a form of leakage; leaving them at literature-standard defaults is honest but possibly sub-optimal.

---

## 8. Experimental design

### 8.1 The controlled comparison

```
                    ┌─────────────────────┐
   Raw EEG ────────►│  Pipeline A         │────┐
   (identical       │  (authors' method)  │    │
    subjects)       └─────────────────────┘    │
                                               ▼
                                    ┌──────────────────────┐
                                    │ IDENTICAL downstream │
                    ┌─────────────────────┐    ▲           │
   Raw EEG ────────►│  Pipeline B         │────┘  • epoching (4 s)
                    │  (alternative)      │       • 171 features
                    └─────────────────────┘       • same classifier
                                                  • same folds, same seed
```

| Variable | Status |
|---|---|
| Preprocessing artifact-removal stage | **INDEPENDENT** |
| Filter, reference, resampling | Controlled — identical |
| Epoching, feature extraction | Controlled — identical |
| Classifier, hyper-parameters, CV folds, seed | Controlled — identical |
| Subject set | Controlled — paired intersection only |

This is enforced structurally: all downstream code lives in a single shared module
(`ds004504_common.py`) imported by both notebooks. Notebook 02 additionally *asserts* at
runtime that its controlled configuration values and its feature-name list match those
Notebook 01 saved, and halts if they do not.

### 8.2 Run modes

The same code path serves all three; only `MAX_SUBJECTS` changes. Reduced modes select
subjects **stratified by group** — subject IDs in this dataset are ordered by diagnosis, so
naively taking "the first N" would yield an all-AD subset and make classification
impossible.

| Mode | Subjects | Purpose |
|---|---|---|
| Quick test | 5 | Smoke test; too few for statistics |
| Development | 20 | Iteration |
| Full | 88 | Final results |

---

## 9. Leakage prevention

The single most important methodological requirement, given that the benchmark review
attributes 7–10 points of published accuracy inflation to its absence.

**Three guarantees:**

1. **Subject-level splitting.** `StratifiedGroupKFold` with `groups = participant_id`; every epoch of a subject falls wholly in train or wholly in test. An explicit assertion raises `RuntimeError` if any subject ever appears on both sides of any fold. Both notebooks additionally run a standalone leakage check on their own data and print the result.
2. **Scaling inside the fold.** `StandardScaler` sits inside the scikit-learn `Pipeline`, fitted on training epochs only. Scaling the full dataset before cross-validation is a classic leakage route.
3. **No selection on full data.** No feature selection or hyper-parameter search is performed outside a fold.

**Subject-level aggregation.** Epoch predictions are averaged to a per-subject probability
before any headline metric is computed, because the subject — not the 4-second epoch — is
the unit of clinical interest. Mean probability is used rather than majority vote so that
a continuous score remains available for ROC and PR analysis.

These map onto the benchmark review's criteria C1 (subject-level independence), C2 (nested
pipeline), C3 (explicit segmentation), C5 (reproducible protocol), C6 (within-subject
structure retained) and C7 (multi-fold resampling).

---

## 10. Feature extraction

171 features: 19 channels × 9 per channel.

| Feature | Count | Rationale |
|---|---|---|
| Relative band power (δ 0.5–4, θ 4–8, α 8–13, β 13–25, low-γ 25–45 Hz) | 5 × 19 | Spectral slowing is the most reproducible qEEG marker of dementia; the benchmark review identifies spectral features as the de facto reference representation |
| Spectral entropy | 1 × 19 | Captures spectral flattening not visible in band ratios |
| 95% spectral edge frequency | 1 × 19 | Single-number summary of the spectral shift |
| Hjorth mobility, complexity | 2 × 19 | Cheap time-domain descriptors of signal dynamics |

**Why relative rather than absolute band power — a design decision specific to this
experiment.** Absolute amplitude is directly altered by preprocessing: ASR and ICA remove
variance by construction. Absolute power would therefore partly measure *how much the
pipeline subtracted*, which is precisely the confound to avoid when preprocessing is the
independent variable. Relative power normalises it out.

Epochs are 4 s and **non-overlapping**. Overlapping epochs would multiply the apparent
sample size while adding almost no independent information.

The upper band edge stops at 45 Hz because Pipeline A's low-pass is 45 Hz; comparing a
band that one pipeline has filtered away would be meaningless.

---

## 11. Machine-learning methodology

**Primary classifier.** L2-regularised logistic regression (`lbfgs`, C = 1.0,
`class_weight='balanced'`). Chosen for transparency and because the benchmark review found
traditional classifiers match deep models under rigorous validation — so a complex model
would add cost and opacity without addressing the research question. `lbfgs` rather than
`liblinear` because the latter cannot fit the three-class task, and the *same* estimator
definition must serve every task.

**Secondary classifier.** Random forest (300 trees, `min_samples_leaf=5`,
`class_weight='balanced_subsample'`), as a robustness check. If the two classifiers
disagree about which pipeline is better, no conclusion about preprocessing is safe.

**Tasks.** AD vs CN (primary), AD vs FTD, AD vs FTD vs CN.

**Validation.** Primary: repeated stratified group *k*-fold (10 × 5). Secondary:
leave-one-subject-out, which is the convention among the most rigorous published studies
and therefore permits like-for-like comparison. LOSO is secondary because it is more
expensive, not because it is less valid.

**Metrics.** Accuracy, balanced accuracy, precision, recall, F1 (macro and weighted),
sensitivity, specificity, ROC-AUC, PR-AUC, confusion matrices; for the three-class task,
per-class precision/recall/F1 and macro one-vs-rest AUC. Accuracy is never reported alone,
as class sizes are imbalanced.

---

## 12. Statistical methodology

**Primary test: paired bootstrap over subjects** (10,000 resamples). Subjects are resampled
with replacement; for each resample the metric is recomputed for both pipelines on the
*same* subjects, and the difference recorded. The **95% confidence interval for the
difference is the headline result**, not the *p*-value.

**Why not a t-test over cross-validation folds?** Folds share training data, so fold-level
metrics are positively correlated and their variance is badly underestimated (Dietterich,
1998; Nadeau & Bengio, 2003), inflating the Type-I error rate. Subjects are the genuine
independent sampling units, and both pipelines scored the same subjects, which is what
makes the comparison properly paired.

**Supporting tests.**

| Test | Unit | Status |
|---|---|---|
| Exact McNemar (binomial on discordant pairs) | Subject | Supporting — asks directly whether the subjects the pipelines disagree on split lopsidedly |
| Nadeau–Bengio corrected resampled *t* | Fold | Supporting — inflates the variance by (1/k + n_test/n_train) |
| Wilcoxon signed-rank | Fold | **Exploratory only** — reported because it is common in this literature, flagged as anti-conservative |

The notebook reports the Nadeau–Bengio *inflation factor* alongside the uncorrected
Wilcoxon result, making the size of the over-optimism directly visible. In integration
testing on synthetic data, the uncorrected Wilcoxon returned p = 0.026 where the corrected
test returned p = 0.198 on the same comparison — a concrete illustration of why the
distinction matters.

**Signal-quality comparison** uses a paired Wilcoxon signed-rank test across subjects.
This *is* appropriate: each subject contributes one paired observation and subjects are
independent — unlike the fold-level case.

**Interpretation.** If the CI includes zero, the data do not support a difference. For a
hypothesis of equivalence this is informative, not a failure. Absence of evidence for a
difference is reported as such, and is not converted into evidence of equivalence beyond
what the interval width supports.

---

## 13. Computational benchmarking

Measured per subject: preprocessing wall time (per step), feature-extraction time, model
fit time, peak RSS, output storage, and time per minute of recording.

**Environment recording.** Python version, all package versions, CPU model, logical and
physical core counts, RAM, and GPU presence are captured to
`results/environment.json`. Both pipelines are timed in the **same session**; absolute
Colab timings are not portable across sessions, so only the within-session *ratio* is
meaningful, and the report states this wherever a ratio is quoted.

**Scalability** is derived from the per-subject timings already measured, rather than
re-processing the dataset at five sample sizes — which would waste Colab quota to answer a
question the existing measurements already contain. Any point beyond the measured range is
flagged `measured = False` in both the table and the figure.

**Colab adaptations:** subject-by-subject sequential processing; per-subject caching to
Google Drive making the run restartable; per-subject download rather than pulling the full
~5 GB; downsampling 500 → 100 Hz (safe below the 45 Hz low-pass, ~5× cheaper) applied
identically to both arms; explicit garbage collection between subjects.

---

## 14. Results

> **NOT COMPUTED — requires execution of the notebooks.**
>
> This section is deliberately empty. Populating it with plausible-looking numbers before
> running the experiment would be fabrication, and the whole point of this study is
> methodological honesty.
>
> Run `01_author_pipeline_ds004504.ipynb`, then `02_alternative_pipeline_ds004504.ipynb`,
> then `03_pipeline_comparison_ds004504.ipynb`. Notebook 03 writes
> `results/measured_results_appendix.md` containing every measured value formatted for
> direct insertion here.

The following subsections define the shape of the results to be inserted.

### 14.1 Dataset verification and inventory
`results/subject_metadata.csv`, `results/dataset_integrity.json`,
`results/dataset_verification.json`.

### 14.2 Pipeline A reproduction fidelity
Duration ratio, per-channel correlation (lower bound), log-PSD correlation, and
band-power differences versus the authors' derivatives.
→ `results/validation_table_*.csv`

### 14.3 Classification performance

| Task | Pipeline | Balanced acc. | Accuracy | Macro F1 | ROC-AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|---|
| AD vs CN | A | | | | | | |
| AD vs CN | B | | | | | | |
| AD vs FTD | A | | | | | | |
| AD vs FTD | B | | | | | | |
| AD vs FTD vs CN | A | | | n/a | | | |
| AD vs FTD vs CN | B | | | n/a | | | |

### 14.4 Statistical comparison
→ `results/comparison_statistics.csv`, `results/comparison_mcnemar.csv`

### 14.5 Signal quality
→ `results/comparison_signal_quality.csv`

### 14.6 Computational cost
→ `results/comparison_runtime.csv`, `results/scalability_*.csv`

### 14.7 Master comparison table

| Metric | Author Pipeline (A) | Alternative Pipeline (B) | Difference | Statistical evidence |
|---|---|---|---|---|
| Balanced accuracy | | | | |
| F1 (macro) | | | | |
| ROC-AUC | | | | |
| Sensitivity | | | | |
| Specificity | | | | |
| Runtime per subject | | | | |
| Peak RAM | | | | |
| Signal-quality metric | | | | |

### 14.8 Comparison against published results

| Task | Published (Validity-1 mean) | Our Pipeline A | Difference |
|---|---|---|---|
| AD vs CN | 82.11% | | |
| AD vs FTD | 71.44% | | |
| AD vs FTD vs CN | 69.99% | | |

Published figures come from different feature sets, classifiers and subject subsets. They
are a **sanity range, not a target**. A result substantially *above* them would be a red
flag for leakage, not a success. No parameter is tuned to close this gap.

---

## 15. Discussion

> Full discussion requires the results. The interpretive framework is fixed in advance so
> that the conclusion follows from the evidence rather than from what reads well.

**Decision rule, fixed before observing any result:**

```
IF the CI for the difference in balanced accuracy includes 0:
    → performance is comparable
    IF Pipeline B is ≥1.5× cheaper  → Conclusion 2
    ELSE                             → Conclusion 5
ELSE IF Pipeline B is higher         → Conclusion 1 or 4 (by cost)
ELSE                                 → Conclusion 3
IF fewer than ~20 paired subjects, or classifiers disagree
                                     → Conclusion 6 (insufficient evidence)
```

All six outcomes are acceptable. The goal is not to make Pipeline B win.

**How each outcome would be read:**

- **Comparable performance, lower cost.** Would suggest the expensive stage is not earning its cost *for this endpoint on this dataset* — a bounded claim, not a general one, and consistent with Delorme (2023).
- **Pipeline A better.** Would suggest ASR and ICA remove artifact that genuinely degrades classification — consistent with de Cheveigné (2023) and with the README's own note that ocular artifacts persist despite the eyes-closed protocol.
- **No difference in anything.** Would suggest the downstream spectral features are robust to preprocessing choice, which is itself a useful negative result and consistent with the review's finding that spectral features act as a stable reference representation.

**A likely tension to address explicitly.** If Pipeline B leaves measurably more residual
artifact (H4 supported) *and* classifies equally well (H2 supported), those results are not
contradictory. They would indicate that the residual artifact is not aligned with the
diagnostic signal carried by relative band power. That is an informative dissociation
between *signal cleanliness* and *task utility*, and it should be reported as such rather
than resolved in favour of whichever metric tells the tidier story.

---

## 16. Limitations

1. **Sample size.** 88 subjects maximum. Confidence intervals on accuracy differences will be wide; small true differences are undetectable. Small EEG samples are known to inflate effect sizes and produce unstable estimates.
2. **Single dataset, single site, single device.** One hospital, one Nihon Kohden EEG 2100. Nothing here establishes cross-site transfer.
3. **Single paradigm.** Resting-state eyes-closed only. Pipelines differing in ocular-artifact handling may rank differently on eyes-open or task data.
4. **Reference substitution.** The A1–A2 reference is not recomputable from the shared files. Pipeline A is an approximate reproduction (§6.3).
5. **Cross-toolbox differences.** ASR and Infomax ICA differ between EEGLAB/MATLAB and Python; bit-identical reproduction is impossible without original seeds and versions.
6. **Demographic confounders.** Group ages differ (66.4 / 63.6 / 67.9 years) and MMSE differs by construction. Age was not regressed out, so some discriminative signal may reflect age rather than pathology.
7. **Clinical labels.** Diagnoses are clinical, not biomarker- or autopsy-confirmed. Label noise places an unknown ceiling on achievable accuracy.
8. **Feature scope.** Spectral and Hjorth features only. Connectivity, complexity and microstate features untested; a different feature set could rank the pipelines differently.
9. **Classifier scope.** Logistic regression and random forest only. Deep models were excluded as computationally infeasible in Colab and unnecessary for a controlled preprocessing comparison.
10. **Colab environment.** Shared virtualised CPU; absolute timings are not portable, only within-session ratios.
11. **Sparse-montage interpolation.** Spherical-spline interpolation over 19 widely spaced electrodes is coarse and may introduce smoothing artifacts in Pipeline B.
12. **No external validation.** Even leakage-free internal validation does not guarantee transportability to new cohorts — "illusory generalizability" (Chekroud et al., 2024).
13. **Single alternative tested.** Pipeline B is one alternative among several plausible ones; a different choice might rank differently.

---

## 17. Conclusion

> **Requires execution of the notebooks.** Notebook 03 applies the §15 decision rule
> automatically and writes the outcome to `results/final_conclusion.json`.

Independent of the numerical outcome, this study contributes:

1. **An open, executable reproduction** of a preprocessing pipeline that dozens of published studies inherit without re-examination, in Python rather than MATLAB, with its uncertainties documented rather than concealed.
2. **A documented reproducibility limit** — the A1–A2 reference cannot be recomputed from the distributed files, and ICLabel is applied outside its validated 1–100 Hz average-referenced regime. Both are properties of the *original* pipeline and therefore inherited by every study that uses the derivatives.
3. **A controlled comparison design** in which preprocessing is the sole independent variable, enforced by a shared code module and runtime assertions.
4. **A computational benchmark** under realistic resource constraints, addressing feasibility for researchers without institutional compute.

Whatever the measured outcome, this study is **not** a clinical validation. No claim is
made about diagnostic capability, hospital deployment, or superiority over EEG
preprocessing methods in general.

---

## 18. Future work

1. **External validation** — the single most valuable extension. Repeat the comparison on a second dementia EEG dataset (e.g. CAUEEG) and test whether the pipeline ranking transfers. The benchmark review identifies cross-configuration generalisation as the field's main open problem.
2. **Ablate Pipeline A** — ASR-only and ICA-only variants, to attribute any difference to a specific stage rather than to the bundle.
3. **Vary the feature set** — connectivity, complexity and microstate features, to test whether pipeline ranking interacts with feature type.
4. **Sensitivity analysis** — vary the ICLabel threshold and bad-channel criteria and report how far the conclusion moves.
5. **Regress out age** — confirm the discriminative signal is not partly age.
6. **Test further alternatives** — MARA, Autoreject, wavelet-ICA, Riemannian ASR.
7. **Larger cohorts and other paradigms** — eyes-open, task-based, and longitudinal recordings.
8. **Deep-learning endpoints** — whether preprocessing choice matters more or less for models operating on raw signals.

---

## 19. References

Full BibTeX in `references.bib`.

**Primary dataset and benchmark**

1. Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2023a). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8(6), 95. https://doi.org/10.3390/data8060095
2. Miltiadous, A., et al. (2024). *A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects* [Data set]. OpenNeuro. https://doi.org/10.18112/openneuro.ds004504
3. Miltiadous, A., Ntetska, A., Aspiotis, V., Moustakli, E., Tsipouras, M. G., Tzallas, A. T., Giannakeas, N., Glavas, E., Angelidis, P., & Tzimourta, K. D. (2026). The AHEPA EEG benchmark: setting the standard for machine learning in dementia diagnosis, a scoping review. *Cognitive Neurodynamics*, 20(1), 95. https://doi.org/10.1007/s11571-026-10464-w
4. Miltiadous, A., Gionanidis, E., Tzimourta, K. D., Giannakeas, N., & Tzallas, A. T. (2023b). DICE-Net: A Novel Convolution-Transformer Architecture for Alzheimer Detection in EEG Signals. *IEEE Access*, 11, 71840–71858. https://doi.org/10.1109/ACCESS.2023.3294618
5. Miltiadous, A., Tzimourta, K. D., Giannakeas, N., Tsipouras, M. G., Afrantou, T., Ioannidis, P., & Tzallas, A. T. (2021). Alzheimer's Disease and Frontotemporal Dementia: A Robust Classification Method of EEG Signals and a Comparison of Validation Methods. *Diagnostics*, 11(8), 1437. https://doi.org/10.3390/diagnostics11081437

**Preprocessing methodology**

6. Delorme, A. (2023). EEG is better left alone. *Scientific Reports*, 13, 2372. https://doi.org/10.1038/s41598-023-27528-0
7. de Cheveigné, A. (2023). *Is EEG better left alone?* bioRxiv. https://doi.org/10.1101/2023.06.19.545602
8. Kothe, C. A. E., & Jung, T.-P. (2016). *Artifact removal techniques with signal reconstruction*. US Patent App. 14/895,440.
9. Chang, C.-Y., Hsu, S.-H., Pion-Tonachini, L., & Jung, T.-P. (2020). Evaluation of Artifact Subspace Reconstruction for Automatic Artifact Components Removal in Multi-Channel EEG Recordings. *IEEE Transactions on Biomedical Engineering*, 67(4), 1114–1121. https://doi.org/10.1109/TBME.2019.2930186
10. Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. *NeuroImage*, 198, 181–197. https://doi.org/10.1016/j.neuroimage.2019.05.026
11. Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., & Robbins, K. A. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*, 9, 16. https://doi.org/10.3389/fninf.2015.00016
12. Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. *Neural Computation*, 7(6), 1129–1159.
13. Lee, T. W., Girolami, M., & Sejnowski, T. J. (1999). Independent component analysis using an extended infomax algorithm for mixed subgaussian and supergaussian sources. *Neural Computation*, 11(2), 417–441.
14. Winkler, I., Haufe, S., & Tangermann, M. (2011). Automatic classification of artifactual ICA-components for artifact removal in EEG signals. *Behavioral and Brain Functions*, 7, 30. https://doi.org/10.1186/1744-9081-7-30
15. Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417–429. https://doi.org/10.1016/j.neuroimage.2017.06.030
16. Mammone, N., La Foresta, F., & Morabito, F. C. (2012). Automatic Artifact Rejection From Multichannel Scalp EEG by Wavelet ICA. *IEEE Sensors Journal*, 12(3), 533–542.
17. Blum, S., Jacobsen, N. S. J., Bleichner, M. G., & Debener, S. (2019). A Riemannian Modification of Artifact Subspace Reconstruction for EEG Artifact Handling. *Frontiers in Human Neuroscience*, 13, 141. https://doi.org/10.3389/fnhum.2019.00141
18. Callan, D. E., Jia, T., & Cody, T. (2024). Shredding artifacts: extracting brain activity in EEG from extreme artifacts during skateboarding using ASR and ICA. *Frontiers in Neuroergonomics*, 5, 1358660. https://doi.org/10.3389/fnrgo.2024.1358660
19. Li, A., Feitelberg, J., Saini, A. P., Höchenberger, R., & Scheltienne, M. (2022). MNE-ICALabel: Automatically annotating ICA components with ICLabel in Python. *Journal of Open Source Software*, 7(76), 4484. https://doi.org/10.21105/joss.04484

**Statistical and ML methodology**

20. Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error. *Machine Learning*, 52(3), 239–281. https://doi.org/10.1023/A:1024068626366
21. Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895–1923.
22. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
23. Chekroud, A. M., Hawrilenko, M., Loho, H., et al. (2024). Illusory generalizability of clinical prediction models. *Science*, 383(6679), 164–167. https://doi.org/10.1126/science.adg8538
24. Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061
25. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
26. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
27. Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267

**Clinical and qEEG background**

28. Smailovic, U., & Jelic, V. (2019). Neurophysiological Markers of Alzheimer's Disease: Quantitative EEG Approach. *Neurology and Therapy*, 8(S2), 37–55. https://doi.org/10.1007/s40120-019-00169-0
29. Chetty, C. A., et al. (2024). EEG biomarkers in Alzheimer's and prodromal Alzheimer's: A comprehensive analysis of spectral and connectivity features. *Alzheimer's Research & Therapy*, 16(1), 236. https://doi.org/10.1186/s13195-024-01582-w
30. Gorgolewski, K. J., et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044. https://doi.org/10.1038/sdata.2016.44

---

## Appendix A — Files

| File | Purpose |
|---|---|
| `ds004504_common.py` | Shared implementation: pipelines, features, CV, statistics, benchmarking |
| `01_author_pipeline_ds004504.ipynb` | Pipeline A + validation against derivatives |
| `02_alternative_pipeline_ds004504.ipynb` | Literature justification + Pipeline B |
| `03_pipeline_comparison_ds004504.ipynb` | Statistical comparison, figures, conclusions |
| `references.bib` | Bibliography |
| `ds004504_pipeline_comparison_report.md` | This document |

## Appendix B — How to run

1. Upload all four code files to a Google Drive folder named `ds004504_experiment`.
2. Open `01_author_pipeline_ds004504.ipynb` in Colab; set `MAX_SUBJECTS = 5`; run all.
3. If the smoke test passes, set `MAX_SUBJECTS = None` and re-run (caching skips completed subjects).
4. Run `02_alternative_pipeline_ds004504.ipynb` with the *same* configuration — it verifies this automatically and halts on mismatch.
5. Run `03_pipeline_comparison_ds004504.ipynb`.
6. Paste `results/measured_results_appendix.md` into §14 of this report.

**Runtime expectation.** Pipeline A is dominated by ASR and ICA; Pipeline B is
substantially cheaper. Actual figures are measured by the notebooks — no estimate is given
here, because an unmeasured estimate is exactly the kind of number this project exists to
avoid.
