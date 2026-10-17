# MedTrust AI
## Reliability-Aware Clinical Decision Routing Using MIMIC-IV

MedTrust AI investigates whether **Explanation Reliability** provides useful information beyond calibrated predictive confidence when identifying potentially unreliable clinical predictions, and whether combining both signals can improve **ACCEPT / REVIEW / DEFER** decision routing.

> **Research status:** This repository contains a research prototype and preliminary experiments. It is not a clinically validated system and must not be used to make real patient-care decisions. Results and conclusions are subject to protocol-alignment checks and further validation.

## Research question

Can an explanation-reliability score identify high-confidence clinical predictions whose explanations are unstable or unfaithful, and does adding this score improve selective-risk behaviour compared with a confidence-only baseline at comparable coverage?

The central hypothesis is that **high predictive confidence does not necessarily imply a reliable explanation**. This is a hypothesis to test, not an assumed result.

## Task and cohort

The frozen research specification defines the task as **patient-level, 24-hour early prediction of in-hospital mortality** for adult ICU patients using MIMIC-IV.

- **Dataset:** MIMIC-IV (the MIMIC-IV Demo is for pipeline development only).
- **Population:** Adults aged 18 years or older.
- **ICU stay:** First ICU stay only.
- **Minimum observation:** At least 24 hours of ICU data.
- **Index time:** ICU admission + 24 hours.
- **Feature window:** The first 24 hours after ICU admission.
- **Target:** In-hospital mortality.
- **Exclusions:** Patients who die before or at the index time.
- **Leakage prevention:** No information from after the index time may be used as a predictor.
- **Unit of analysis:** Patient-level.

MIMIC-IV access is controlled by its data provider and requires completing the applicable credentialing and data-use requirements. Obtain the dataset through its official source; do not commit raw data, patient-level extracts, credentials, or other restricted files to this repository.

## Proposed pipeline

```text
MIMIC-IV
   ↓
Cohort construction and 24-hour feature extraction
   ↓
Leakage checks and patient-level train/validation/test split
   ↓
Candidate models: Logistic Regression, Random Forest,
                  XGBoost, LightGBM, CatBoost
   ↓
Select the best validated model and calibrate its probabilities
   ↓
Calibrated predictive confidence + SHAP explanations
   ↓
Explanation Reliability Score (ERS)
   ↓
Confidence-only baseline vs confidence + ERS
   ↓
ACCEPT / REVIEW / DEFER routing
   ↓
Ablation, robustness, and selective-risk evaluation
```

## Explanation Reliability Score (ERS)

ERS is intended to summarize how dependable an individual prediction's explanation is. The specification calls for validation and analysis of explanation properties such as:

- **Stability:** whether explanations remain similar under controlled perturbations.
- **Feature-ranking consistency:** whether important features remain highly ranked.
- **Sign consistency:** whether attribution directions remain consistent.
- **Faithfulness:** whether explanation-based expectations agree with observed model behaviour under controlled feature interventions.

The exact composite formulation is not assumed to be correct in advance. Candidate formulations, component ablations, and sensitivity analyses must be evaluated on validation data. The chosen formulation and routing thresholds must be frozen before final test evaluation.

## Decision routing

The system combines calibrated confidence with ERS to support three possible review states:

| Predictive confidence | Explanation reliability | Intended route |
|---|---|---|
| High | High | **ACCEPT** |
| High | Low | **REVIEW** |
| Low | High | **REVIEW** |
| Low | Low | **DEFER** |

The terms describe a research decision-support policy, not permission to accept or reject actual clinical care. Thresholds must be selected using validation data, not tuned on the final test set.

## Experimental comparisons

The planned evaluation compares progressively richer systems:

1. **Prediction only:** predictive performance baseline.
2. **Prediction + calibrated confidence:** confidence-only baseline.
3. **Prediction + confidence + SHAP:** explanations presented without ERS-based routing.
4. **Prediction + confidence + ERS:** test the incremental value of explanation reliability.
5. **Full routing system:** prediction, calibrated confidence, SHAP, ERS, and ACCEPT / REVIEW / DEFER routing.

The central experiment examines whether high-confidence/low-ERS cases have a higher error rate than high-confidence/high-ERS cases. Routing is compared with confidence-only baselines at comparable coverage.

Planned evaluation includes discrimination, probability calibration, error detection, subgroup analysis, coverage-risk behaviour, ablations, and robustness/sensitivity analysis. All final thresholds and model-selection choices should be determined on validation data and frozen before test evaluation.

## Repository structure

```text
.
├── configs/
│   ├── ers_selection_val_noise0.1.json
│   ├── ers_selection_val_noise0.25.json
│   ├── feature_columns.txt
│   ├── features.yaml
│   ├── model_params.json
│   ├── primary_model.json
│   ├── routing_val.json
│   └── split_info.txt
├── reports/
│   ├── model_comparison_val.csv
│   ├── feature_missingness.csv
│   ├── leakage_audit_single_feature_auroc.csv
│   ├── ers_incremental_val_noise0.1.csv
│   ├── ers_incremental_val_noise0.25.csv
│   ├── routing_val.csv
│   ├── test_all_metrics.csv
│   └── additional diagnostic tables and plots
├── src/
│   ├── data/          # input checks, item IDs, extraction, cohort, features, split, leakage audit
│   ├── models/        # candidate model training and calibration/model selection
│   ├── reliability/   # SHAP perturbation and ERS analysis
│   ├── routing/       # validation routing and fixed-gate analysis
│   └── evaluation/    # final test evaluation
├── requirements.txt
├── setup_project.py
└── .gitignore
```

Generated data files and model artifacts may be required to run parts of the pipeline but are not committed to this repository. Review each script and configuration before running it.

## Setup

Use Python compatible with the pinned packages in `requirements.txt`. From the repository root, create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place legally accessed MIMIC-IV source files in the locations expected by the data-ingestion scripts. The scripts use paths under `data/`; inspect the scripts and `configs/features.yaml` to confirm the required filenames and schemas before execution. The `data/` directory and Parquet files are excluded from Git by `.gitignore`.

## Pipeline entry points

The numbered scripts reflect the current implementation order. Run them from the repository root and verify each script's input requirements before proceeding.

```text
src/data/01_check_files.py
src/data/02_find_itemids.py
src/data/03_to_parquet.py
src/data/04_cohort.py
src/data/05_features.py
src/data/06_split.py
src/data/07_leakage_audit.py
src/models/08_train_models.py
src/models/09_calibrate_select.py
src/reliability/10_shap_perturb.py
src/reliability/11_ers_incremental.py
src/routing/12_routing_val.py
src/routing/12b_fixed_gate_val.py
src/evaluation/13_final_test.py
```

For example, after confirming the necessary input files exist, run an individual stage from the project root:

```powershell
python src/data/01_check_files.py
```

Do not assume every stage can run from a clean clone without locally prepared data and intermediate artifacts.

## Important protocol-alignment note

The research specification requires features from the **first 24 hours after ICU admission only**. The currently tracked `configs/features.yaml` sets `lab_lookback_hours: 6`, and `src/data/05_features.py` uses that setting to include lab measurements from up to six hours before ICU admission. This currently differs from the frozen specification and must be resolved before treating results as confirmatory.

The saved test outputs have already been inspected during development. If the feature pipeline or ERS-selection procedure is corrected, rerunning on the same test set should be described as a post-hoc evaluation, not as an untouched confirmatory test. Preserve the existing split and report the protocol change transparently.

## Current findings and interpretation

The repository contains preliminary validation and test diagnostics. These should not be interpreted as proof that ERS improves routing. In the experiments reviewed so far, the tested ERS formulations did **not** demonstrate clear incremental benefit over calibrated confidence alone. Correcting the feature-window deviation and the validation uncertainty-interval implementation is necessary before drawing a final conclusion.

The project explicitly allows a negative result: if ERS does not add useful information at practical coverage, that finding should be reported rather than forcing a positive claim.

## Limitations and responsible use

- This is a research prototype, not a medical device or clinical decision system.
- Predictive performance does not establish clinical utility or safety.
- SHAP explanations are not automatically faithful, stable, or clinically meaningful.
- ERS is an experimental measure whose definition and value must be validated.
- MIMIC-IV is a retrospective dataset; results may not generalize to other hospitals or populations.
- Protect patient privacy and comply with the dataset's access and data-use requirements.
- Report negative, null, and sensitivity findings, not only favourable results.

## Reproducibility principles

1. Document cohort definitions, feature windows, and exclusions.
2. Keep patient-level train/validation/test partitions separate.
3. Select models, calibration, ERS formulations, and routing thresholds using validation data.
4. Freeze choices before final test evaluation.
5. Report confidence-only baselines and comparisons at comparable coverage.
6. Record code/configuration versions and any post-hoc analyses.
7. Do not present demo results as final full-dataset research findings.

## License and data

No open-source license is specified in this repository at present. Dataset rights and access conditions remain governed by the dataset provider. Do not redistribute MIMIC-IV data or derived patient-level records.

---

**Project:** MedTrust AI — Model 1  
**Research direction:** Reliability-Aware Clinical Decision Routing  
**Primary dataset:** MIMIC-IV  
**Core question:** Does explanation reliability add useful information beyond calibrated predictive confidence?
