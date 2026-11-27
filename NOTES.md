# Nifty 50 volatility-event classifier

The prediction is whether either intraday extreme will cross the 2% band around the previous close. Both directions count as the same event. The model uses the information available before 09:15 IST and emits one Boolean per stock.

## Inputs and model

The final model is an L2-regularized logistic regression with five inputs. Daily implied volatility converts the stock's annualized option IV to percentage points per day using `vix / sqrt(252)`. One-month IV reflects the stock's typical volatility, the daily-to-monthly ratio captures recent changes, and the supplied smoothed VIX z-score gives F&O market context. The previous-session high–low range divided by its close measures realized volatility. The daily Nifty 50 median implied volatility is used by the decision rule.

Each fit estimates its own medians, 1st/99th percentile clipping bounds and IQR scales. More recent usable training sessions get more weight with a 35-session half-life. The predictor uses the newest of 19 JSON snapshots whose cutoff is strictly earlier than the requested date. During the initial 16 supplied sessions it uses a predefined rule combining 70% implied daily volatility with 30% of the prior range divided by 1.55. That rule flags a blend of at least 2.45 percentage points and uses the available component when one is missing. If both primary inputs are unavailable, the predictor emits `False`.

## Training data

The 4,850 training rows have 1,996 positives (41.2%), no duplicate keys, and no infinite or malformed numeric values after recognizing the supplied missing tokens. Event rates by date ranged from 8% to 94%. The top quintile of daily IV had a 69.3% move rate versus 19.5% in the bottom quintile. The same comparison for the previous-session range was 61.5% versus 23.8%. Stock IV and prior range had a Spearman correlation of 0.42. The two IV z-scores had a correlation above 0.999, and the six-month and one-year IV averages were equal on 80.8% of rows with current IV. Median traded value was ₹4.04 billion. Relative trading value quintiles had much weaker separation than IV or the range.

The 29 sessions with file-wide masking described by the kit account for 1,450 rows. Natural gaps accounted for another 400 missing ranges across eight sessions and 18 missing IV values. A row can still be usable with one primary input. The training fit excludes rows lacking both inputs and uses neither coverage indicators nor any attempt to rebuild masked values. [Feature quintiles](reports/volatility_quintiles.png) · [Market regimes](reports/market_regimes.png) · [Detailed EDA](reports/eda.json)

## Validation and decision rule

I used the first 16 supplied training sessions for warm-up, then made predictions in five-session increments and grouped the development results into four chronological blocks. Every model and preprocessor was fit on earlier eligible rows. The operating threshold for each development increment used out-of-fold predictions from earlier dates, with an initial threshold of 0.74 until at least eight usable OOF sessions were available. The final 18 supplied sessions, 16 September through 30 October 2026, were evaluated with one model fitted through 3 September and one frozen decision rule. An initial data audit viewed coarse 10-session class-rate aggregates across all training dates. The final 18 sessions' row labels and model performance were excluded from candidate selection.

The official score is the square root of the weighted harmonic mean of precision and recall with weights 0.75 and 0.25. Passing requires precision strictly greater than 0.60. I screened precision targets of 0.65, 0.68, 0.70, 0.72, and 0.75, evaluated variability across chronological blocks, and reweighted calm sessions in a stress scenario. The selected policy targeted development precision of 0.68. Learned base thresholds were 0.72–0.74. The holdout base threshold was 0.73, raised to 0.775 whenever the same-morning cross-sectional median daily IV was below 1.70%. The threshold is applied independently to every stock.

### Comparable development results

The comparisons use the same 2,198 usable out-of-fold rows. PR-AUC is computed as average precision.

| Model | Precision | Recall | Score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Original template | 0.547 | 0.434 | 0.717 | 0.527 | 0.711 |
| Implied daily IV threshold | 0.542 | 0.489 | 0.727 | 0.556 | 0.721 |
| Prior range threshold | 0.549 | 0.263 | 0.657 | 0.439 | 0.575 |
| Combined volatility rule | 0.547 | 0.476 | 0.726 | 0.559 | 0.724 |
| Two-feature logistic regression | 0.677 | 0.204 | 0.654 | 0.560 | 0.725 |
| **Selected five-feature logistic regression** | **0.687** | **0.191** | **0.645** | **0.560** | **0.720** |
| Sigmoid-calibrated logistic regression | 0.704 | 0.184 | 0.642 | 0.550 | 0.707 |
| Wider LightGBM | 0.571 | 0.247 | 0.656 | 0.550 | 0.705 |
| Wider CatBoost | 0.587 | 0.267 | 0.672 | 0.555 | 0.711 |
| Logistic and CatBoost blend | 0.625 | 0.214 | 0.650 | 0.549 | 0.712 |

The selected model produced 145 true positives, 66 false positives and 616 false negatives. The 90% trading-date bootstrap interval for precision was 0.576–0.790. Fold scores were 0.781, 0.196, 0.299 and 0.000, so the pooled value depends heavily on the first block. Lowering the final base threshold by 0.03 in a development diagnostic brought precision down to 0.623. The fitted decision rule was kept fixed for the holdout.

The wider feature sets, shallow trees and blends offered no consistent gain under the precision constraint. Several higher raw scores came with precision below the passing bar. Removing rolling features from the broad logistic model changed average precision from 0.545 to 0.540. Removing stock priors or calendar variables slightly improved it. Permuting daily blocks reduced the selected model's PR-AUC by 0.055 for implied-volatility inputs and 0.027 for prior range. Causal sigmoid calibration reduced Brier loss from 0.201 to 0.190 on 1,749 rows. Its operating score and ranking metrics did not improve enough to replace the raw logistic scores. [Development experiments, thresholds and stability checks](reports/development.json)

### Final chronological holdout

The frozen threshold flagged zero stocks across all 900 rows, including the 600 usable rows. Precision, recall and score were all 0 under the grader's convention. Ranking still produced PR-AUC of 0.581 and ROC-AUC of 0.643 on usable rows. The original rule scored 0.705 at 0.617 precision on those same rows. Removing alternating prior historical sessions left final predictions unchanged because the retained model uses the supplied opening record and same-date market median. This result exposes the model's limited transfer across regimes and the cost of a precision-focused threshold. [Holdout details](reports/holdout.json)

### Official practice grader

| Submission | Precision | Recall | Score |
|---|---:|---:|---:|
| Original template | 0.6272 | 0.3372 | 0.7185 |
| Final snapshots | 0.7500 | 0.1052 | 0.5442 |

The final snapshots passed the precision requirement with 210 true positives and 70 false positives. The grader processed all 97 training dates and all 4,850 rows with zero failures or missing predictions in 176.4 seconds total. This practice run uses rolling snapshots and can incorporate a supplied date's labels in later predictions. The chronological holdout uses a single earlier snapshot throughout. Nine automated tests passed, including future invariance, feature parity, exact output coverage and isolated execution. [Official practice results](reports/practice.json)

## Reproduce

Use Python 3.11. Keep the virtual environment outside the assignment repository because the official practice grader copies the submission directory for each date. In PowerShell, use `py -3.11 -m venv ../yms-env` and `..\yms-env\Scripts\Activate.ps1` for the first two commands.

```sh
python3.11 -m venv ../yms-env
source ../yms-env/bin/activate
python -m pip install -r requirements.txt
python prepare_user_historical_info.py --dates data/all_dates.txt --names data/n50_names.txt --out user_historical_info.csv
python train.py
python eda.py
python -m unittest discover -s tests -v
python evaluator/run_evaluation.py .
```

Use `python train.py --stage develop` to stop after writing the candidate configuration and `python train.py --stage finalize` to evaluate that frozen configuration and regenerate the submission snapshots. Inference runs offline with NumPy and pandas. The supplementary file contains all 6,900 date/stock pairs with calendar features. Candidate rolling features use the exchange calendar and the already-lagged information, and stock priors use only strictly earlier eligible labels. The official scorer and evaluator are unchanged. The official test dates remain unscored. Codex assisted with implementation, experiments and documentation.
