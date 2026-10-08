# Technical Crash Course — Everything in the Pipeline, Explained

Purpose: so you can field any technical question about the pipeline confidently, even if it's not directly covered in the reports. Written assuming you know ML/DL basics (what training/testing is, what a neural net is) but want the *specifics* of every technique used here.

Sections 1–8 cover the pipeline built in Phases 1–2. **Sections 9–14 cover Phase 3** (interpretability, final evaluation with statistical testing, and robustness of all five models), ending with likely examiner questions and answers. Section 15 is the glossary.

---

## 1. Audio & Signal Processing Basics

**Sampling rate (8 kHz).** Audio is a continuous wave; a computer stores it as discrete samples per second. 8 kHz means 8,000 samples/second. This is the standard rate for telephone-quality audio (landline/mobile voice calls use 8 kHz — it's enough to capture human speech intelligibly, roughly up to 4 kHz of frequency content, per the Nyquist theorem: max frequency captured = half the sampling rate). Music/studio recordings use 44.1–48 kHz. I deliberately downsample everything to 8 kHz because that's what a real banking call actually sounds like — training on studio-quality 44 kHz audio and testing on 8 kHz phone audio would be an unrealistic mismatch.

**G.711 codec.** This is the actual audio compression codec used on the global telephone network (both landline and most VoIP). It's a "µ-law" (mu-law) companding scheme — it compresses the dynamic range of the signal non-linearly (more resolution for quiet sounds, less for loud ones, mimicking human hearing sensitivity) to fit into 8 bits/sample instead of 16. I encode and then decode through G.711 to introduce the exact quantization artifacts a real phone call would have. Why this matters for the project: TTS systems are usually evaluated on clean, uncompressed audio, but a fraud attack over a bank's phone line would go through this codec — so my evaluation reflects the real threat model.

**Why 3–8 second clips?** That's the typical length of a spoken transaction-confirmation phrase ("Yes, I confirm this transfer of five thousand rupees") — not a long conversation. Longer utterances give models more signal to work with, so testing on realistically short clips is a harder, more honest test.

**SNR (Signal-to-Noise Ratio), measured in dB.** Ratio of the power of the actual speech signal to the power of background noise, in decibels (logarithmic scale). Higher dB = cleaner audio. 20 dB SNR is a fairly clean call; 5 dB SNR is quite noisy (think: someone on a call from a crowded street). I added white Gaussian noise at 5/10/15/20 dB to simulate this. "White Gaussian noise" = random noise with equal energy at every frequency, statistically Gaussian-distributed amplitude — a standard, tractable noise model, though not a perfect stand-in for real background chatter (a limitation I'd acknowledge if asked).

---

## 2. Handcrafted Features (the 257-dimensional vector)

These are numbers computed directly from the audio waveform using signal-processing formulas — no learning involved, just math. Each feature is computed frame-by-frame and then summarised over the whole clip (mean and standard deviation), which is what turns a variable-length clip into one fixed 257-number vector. The MFCC and spectral features use librosa's default frame: 2048 samples (256 ms at 8 kHz), moving forward 512 samples (64 ms) at a time. That is longer than the textbook 25 ms / 10 ms speech frame — if asked, be honest: it trades time resolution for frequency resolution, and since only clip-level means/stds are kept, the effect on the summary statistics is modest (but it does smooth the deltas).

**Where 257 comes from:** 40 MFCCs × {static, Δ, ΔΔ} × {mean, std} = 240, + 9 spectral (centroid, bandwidth, rolloff, ZCR — each mean & std — plus HNR) + 5 pitch (F0 mean/std/min/max + voiced-frame ratio) + 3 phase = **257**.

**MFCC (Mel-Frequency Cepstral Coefficients) — 40 coefficients × {static, Δ, ΔΔ} × {mean, std} = 240 dims.**
- Take the audio frame → FFT to get its frequency spectrum → warp the frequency axis onto the **Mel scale** (a scale that matches human pitch perception — we're more sensitive to differences at low frequencies than high) → take the log of the energy in each Mel band → apply a **Discrete Cosine Transform (DCT)** to decorrelate the bands into a compact set of coefficients.
- Intuitively: MFCCs describe the *shape* of the vocal tract's frequency response — the "timbre" of the sound — in a way that closely matches how humans perceive it and that's compact (40 numbers instead of a full spectrum).
- **Delta (Δ) and delta-delta (ΔΔ):** the first and second derivatives (rate of change, and rate of change of that rate) of the MFCCs over time. If you have MFCCs at frame *t*, delta captures how they're changing frame-to-frame; delta-delta captures acceleration of that change.
- **Why deltas turned out to be the most important features in my Random Forest analysis:** natural human articulation has continuous, somewhat irregular momentum — your vocal tract is a physical system with inertia. TTS vocoders (the neural network that turns a spectrogram back into a waveform) tend to produce smoother, more mechanically regular transitions between frames, because they're generating frame-by-frame from a model rather than a physical articulator. So the *rate of change* of the spectral shape is a more reliable "synthetic vs real" signal than the static shape itself.

**Spectral centroid.** The "center of mass" of the frequency spectrum — literally, a weighted average of frequencies, weighted by their energy. A brighter, higher-pitched sound has a higher centroid. Lower in synthetic speech on average, per my data — vocoders tend to smooth out high-frequency detail.

**Spectral bandwidth.** How spread out the spectrum is around the centroid (like a standard deviation of frequency content). Narrower in vocoder output = energy concentrated more tightly, again a smoothing effect.

**Spectral rolloff.** The frequency below which some percentage (typically 85%) of the total spectral energy is contained. Tells you where the "top" of the meaningful signal is. Found to be about 764 Hz lower on average in synthetic speech — vocoders under-represent high-frequency energy.

**Zero-crossing rate (ZCR).** How many times per frame the waveform crosses zero amplitude. High ZCR correlates with noisy/unvoiced sounds (like "s", "f"); low ZCR with voiced, periodic sounds (vowels). A rough proxy for how "noisy vs. tonal" a sound is.

**Harmonic-to-Noise Ratio (HNR).** Ratio of energy in the harmonic (periodic, voiced) part of the signal vs. the noise-like part. Real voiced speech has some natural noise mixed in (breathiness, imperfect vocal fold vibration); TTS vocoders tend to produce unnaturally clean, regular harmonic structure — higher and more consistent HNR.

**F0 / Pitch (fundamental frequency).** The rate of vocal fold vibration — what we perceive as "pitch." Computed here as mean/std/min/max across the utterance. Captures prosodic naturalness — real speech has more pitch variability and micro-instability than some synthetic speech.

**Phase difference features.** Most of the features above are about the *magnitude* spectrum (how much energy at each frequency) — phase (the timing/alignment of each frequency component) is usually thrown away. But vocoders (especially older/simpler ones) don't always reconstruct phase relationships perfectly — they can introduce "phase coherence artifacts" that are invisible if you only look at magnitude. This was directly motivated by a paper (Patel & Patil, 2015) showing phase-spectrum features catch vocoder artifacts that MFCCs alone miss. It ranked 16th in importance — a real but secondary signal.

**Why standardize features (zero mean, unit variance)?** SVM and gradient-based methods are sensitive to feature scale — a feature ranging 0–10,000 would dominate one ranging 0–1 even if less informative. Standardizing puts every feature on equal footing before training.

---

## 3. Log-Mel Spectrograms (deep learning input)

Instead of collapsing the spectrum into a handful of numbers (like MFCCs), for the deep learning models I feed in something closer to a raw picture: a **128-bin log-Mel spectrogram** — the Mel-scaled spectrum (128 frequency bands covering 0–4,000 Hz) computed with a 512-sample (64 ms) window every 128 samples (**16 ms**), log-compressed (because energy differences are perceived logarithmically, and it also compresses the numeric range), and normalized per clip to zero mean / unit variance. This creates a 2D image: one axis is time, one is frequency (Mel-scaled), and pixel intensity is log-energy.

**Important detail (it matters for Phase 3):** the image is fixed at 128×128 by *cropping* to the first 128 time frames — 128 × 16 ms ≈ **the first 2.05 seconds of each clip**. It is not resized. So the deep models only ever see the opening ~2 s of each 8 s clip, *including any leading silence before the speaker starts*. (Clips shorter than that would be padded with −80 dB "silence", but all processed clips are 8 s.) Keep this in mind when reading the saliency maps in Section 11.

**Why not just use MFCCs for deep learning too?** MFCCs already threw away a lot of detail via the DCT compression step — that's fine for a classical ML model with a fixed, engineered feature set, but a CNN can learn its *own* useful representations directly from the richer, less-compressed Mel spectrogram, which is why deep learning models conventionally use spectrograms rather than MFCCs as input.

---

## 4. Classical ML Models

**SVM (Support Vector Machine) with RBF kernel.**
- Core idea: find the decision boundary (hyperplane) that maximizes the margin (distance) between the two classes.
- **RBF (Radial Basis Function) kernel:** SVMs can only draw a straight-line boundary in the original feature space; a kernel implicitly maps the data into a higher-dimensional space where a straight boundary *can* separate classes that aren't linearly separable in the original space. RBF is essentially a similarity measure based on distance — points close together in feature space get high similarity, far apart get low similarity, decaying like a Gaussian.
- **C = 100 (regularization parameter):** controls the trade-off between a wide margin (simpler, more generalizable boundary) and correctly classifying every training point (risk of overfitting). Higher C = less tolerance for misclassified training points, tighter fit. The grid searched C ∈ {0.1, 1, 10, 100} and picked 100 — the saved model (`outputs/models/svm.joblib`) uses C=100. ⚠️ The Phase 2 report says C=10/gamma="auto"; the saved model says otherwise, so quote **C=100, gamma="scale"** and correct the report.
- **γ (gamma) = "scale":** controls how far the influence of a single training point reaches in the RBF kernel. Small gamma = smooth, far-reaching influence (simpler boundary); large gamma = each point only influences its immediate neighborhood (can overfit). "scale" sets γ = 1 / (n_features × variance of the data); since the features are standardised (variance 1), that is ≈ 1/257 — practically the same as "auto".
- **Support vectors:** the trained SVM keeps 2,104 support vectors (859 genuine, 1,245 synthetic) — the training points that sit on or inside the margin. Every prediction is a weighted sum of kernel similarities to all 2,104, which is why SVM SHAP is slow (Section 10).

**Random Forest.**
- An ensemble of many decision trees (300 in my setup), each trained on a random subset of the data (bootstrap sampling) and a random subset of features at each split. Final prediction = majority vote across all trees.
- Why it works: individual decision trees overfit easily; averaging many "weakly correlated" trees cancels out their individual errors while keeping their collective signal — this is the general "bagging" (bootstrap aggregating) principle.
- **Unlimited depth:** each tree is allowed to grow until leaves are pure (or some minimum sample count) — depth isn't artificially capped, since the ensemble averaging already controls overfitting.
- **Feature importance:** Random Forest gives you a free-standing importance score per feature, computed from how much each feature reduces impurity (Gini impurity — a measure of class mixing) across all the splits that use it, averaged across all trees. This is what produced the delta-MFCC ranking. (Caveat if asked: this is *not* SHAP — it's a simpler, model-specific importance measure, and it can be biased toward high-cardinality/continuous features. SHAP is the more rigorous, model-agnostic version — done in Phase 3, and it agrees with this ranking: ρ = 0.94, see Section 10.)

**Gradient Boosting.**
- Also an ensemble of decision trees (200, in my setup), but built *sequentially* rather than independently: each new tree is trained specifically to correct the errors (residuals) of the ensemble so far, weighted by a **learning rate (0.1)** that controls how much each new tree is allowed to contribute (small steps = more stable, needs more trees).
- Contrast with Random Forest: Random Forest reduces variance by averaging independent trees; Gradient Boosting reduces bias by iteratively focusing on mistakes. In my results, Gradient Boosting turned out to be the most noise-robust classical model — likely because its sequential error-correction process builds a more calibrated decision boundary near the noisy/ambiguous region, though Random Forest and SVM had lower clean-condition EER.

**SMOTE (Synthetic Minority Oversampling Technique).**
- My dataset is imbalanced — roughly 1 genuine : 3.7 synthetic clips (2,656 genuine vs 9,884 synthetic in training; ~79% synthetic). If you train naively on imbalanced data, the model can get high accuracy by just always predicting the majority class.
- SMOTE generates *synthetic* new minority-class (genuine) examples by interpolating between existing minority examples and their nearest neighbors in feature space — not just duplicating them, which would just overweight existing points without adding new information.
- `class_weight='balanced'` is a complementary approach used simultaneously: it reweights the loss function so mistakes on the minority class count more, without changing the actual data.

**Stratified 5-fold cross-validation + grid search.**
- Grid search = systematically trying combinations of hyperparameters (like C and gamma for SVM) and picking the best-performing combination.
- 5-fold CV = split the training data into 5 chunks; train on 4, validate on the 5th, rotate through all 5 combinations, average the validation score. This gives a more reliable performance estimate than a single train/validation split, since every data point gets used for both training and validation across the 5 rounds.
- "Stratified" = each fold preserves the same class ratio as the full dataset — important here given the ~1:3.7 imbalance, so no fold accidentally ends up with too few genuine examples.
- Optimization objective = **ROC-AUC** (see metrics section) rather than accuracy, because accuracy is misleading under class imbalance.

**Decision threshold at the EER operating point.** A classifier outputs a *probability* or *score*, not a hard yes/no — you need to pick a cutoff. Instead of the default 0.5 cutoff, I set the threshold to whichever point on the validation set makes the false-acceptance rate equal to the false-rejection rate — that's the EER-defining threshold, and it's the standard operating point convention in biometrics/anti-spoofing (rather than optimizing for accuracy or F1 directly).

---

## 5. Deep Learning Models

**CNN (Convolutional Neural Network) basics.**
- **Convolution:** slide a small learnable filter (e.g., 3×3) across the input, computing a dot product at each position. Each filter learns to detect a specific local pattern (an edge, a texture, in our case a specific time-frequency artifact shape). Many filters per layer = many different patterns detected in parallel.
- **Batch normalization:** normalizes the activations within each mini-batch during training (zero mean, unit variance, then a learned rescale). Stabilizes and speeds up training, and acts as mild regularization.
- **ReLU (Rectified Linear Unit):** activation function, `f(x) = max(0, x)`. Simple, computationally cheap, avoids the "vanishing gradient" problem that older activations (sigmoid/tanh) suffer from in deep networks.
- **Max-pooling:** downsamples by taking the maximum value in each small local window (e.g., 2×2), reducing spatial resolution while keeping the strongest activations. Makes the representation somewhat translation-invariant and reduces computation for later layers.
- **Global average pooling (GAP):** instead of flattening the final feature maps into a huge vector before the classifier, GAP averages each feature map down to a single number. Drastically reduces parameters (no giant flatten→dense layer) and forces the network to learn spatially-global features — but the trade-off (relevant to why CNN-LSTM does better) is that GAP throws away *where in time* something happened, collapsing the whole clip into one static summary.
- **Dropout (p=0.3):** during training, randomly zero out 30% of neurons in a layer for each batch. Prevents the network from over-relying on any specific neuron/pathway, a standard regularization technique against overfitting.
- **4 conv blocks (32→64→128→256 channels):** each successive block doubles the number of filters, following the common CNN design pattern of building up more abstract, higher-level features (of which there are more possible types) as spatial resolution shrinks.

**CNN-LSTM — what changes and why.**
- Same convolutional front-end, but pooling is done *only across the frequency axis*, never the time axis — so instead of collapsing the whole clip to one vector (like GAP does), you end up with a *sequence* of feature vectors, one per time step, each summarizing the frequency content at that moment.
- **LSTM (Long Short-Term Memory):** a recurrent neural network variant designed to process sequences while retaining relevant information over many time steps (via internal "gates" that control what to remember, forget, and output at each step) — solves the vanishing-gradient problem that plain RNNs have over long sequences.
- **Bidirectional:** two LSTMs run over the sequence — one forward in time, one backward — and their outputs are concatenated at each step. This lets the model use both past and future context when interpreting any given moment (useful since we have the whole clip available at inference time, not doing real-time streaming).
- **Why this matters for the result:** prosody, vocoder smoothing, and articulatory dynamics are fundamentally about *how things change over time* — exactly what GAP discards and what the LSTM is built to capture. That's my explanation for the CNN → CNN-LSTM performance jump (EER 0.0081 → 0.0035 on test).

**Training details.**
- **Adam optimizer:** an adaptive-learning-rate gradient descent variant that keeps a running estimate of both the gradient and its variance (first and second moments) per parameter, adjusting the effective step size for each parameter individually. Faster and more robust than plain SGD for most deep learning tasks; the de facto default optimizer.
- **Weight decay (1e-4):** L2 regularization — penalizes large weights, discourages overfitting.
- **Cosine annealing (learning rate schedule):** the learning rate follows a cosine curve, starting high and smoothly decreasing to near-zero over the 50 epochs. Lets the model take large exploratory steps early and fine, precise steps late in training, generally outperforming a flat or step-wise learning rate schedule.
- **Weighted cross-entropy loss:** cross-entropy is the standard classification loss (penalizes confident-wrong predictions heavily via a log term); "weighted" means the minority class's loss contributes more, addressing the ~1:3.7 imbalance the same way `class_weight='balanced'` does for the classical models.
- **WeightedRandomSampler:** rather than (or in addition to) reweighting the loss, this controls *which examples get sampled into each mini-batch* — oversampling the minority class so batches are roughly balanced by construction, giving the model more frequent exposure to genuine examples.
- **Mixed precision (float16 autocast):** most computation is done in 16-bit floating point instead of 32-bit, which is faster and uses less GPU memory, with some operations kept in 32-bit for numerical stability. This is what made training feasible on a 6.4 GB laptop GPU.
- **Early stopping (patience 10):** stop training if validation EER hasn't improved for 10 consecutive epochs, and keep the best checkpoint — protects against overfitting from training too long.

---

## 6. Evaluation Metrics

**Accuracy.** Fraction of correct predictions. Misleading under class imbalance (predicting "synthetic" for everything gets ~78% accuracy on this test set without learning anything).

**Precision.** Of everything predicted "synthetic," what fraction actually was synthetic? High precision = few false alarms.

**Recall.** Of everything that actually was synthetic, what fraction did the model catch? High recall = few missed attacks.

**F1-score.** Harmonic mean of precision and recall — a single number balancing both, useful when you care about both false alarms and missed detections roughly equally.

**ROC curve / AUC.** ROC plots True Positive Rate vs. False Positive Rate as you sweep the decision threshold from 0 to 1. AUC (Area Under Curve) summarizes this into one number: probability that the model ranks a random positive example higher than a random negative example. AUC = 1.0 is perfect; 0.5 is random guessing. Threshold-independent, which is why it's a good optimization target during grid search.

**EER (Equal Error Rate) — the primary metric.** The point on the ROC curve (equivalently, the DET curve) where **False Acceptance Rate = False Rejection Rate**. In biometrics terms: False Acceptance = letting a spoofed/synthetic voice through as genuine (security failure); False Rejection = rejecting a genuine speaker as fake (usability failure/customer friction). EER finds the single threshold where these two error types are balanced — the standard because it doesn't require choosing an arbitrary trade-off between security and convenience, and it's comparable across systems and papers. Lower EER = better.

**Confusion matrix.** A 2×2 table (for binary classification) of actual class vs. predicted class: true positives, true negatives, false positives, false negatives. Gives you the raw counts behind precision/recall/accuracy — useful for sanity-checking that a metric isn't hiding a lopsided error pattern.

---

## 7. Data Splits & Generalization

**Speaker-independent split.** No speaker appears in more than one of train/val/test. Why it matters: if the same speaker's voice appeared in both train and test, the model could partly learn to recognize *that speaker's* voice characteristics rather than genuinely general genuine-vs-synthetic cues — an easy way to get inflated, misleading numbers.

**System-independent split (for synthetic data).** Similarly, the 70/15/15 split is applied *within* each TTS source so all sources are proportionally represented in every split — but see below for the one deliberate exception.

**Held-out generalization test (Edge-TTS).** Microsoft Edge-TTS is *entirely* excluded from train and validation — zero exposure during training. This directly tests **cross-system generalization**: can the model catch a synthetic speech attack from a TTS system it has literally never seen a single example of? This is the more realistic real-world scenario (an attacker could use any TTS system, not necessarily one in your training data) and is a stronger test than the standard held-out test set (which, despite being "unseen data," still comes from TTS systems the model *did* train on, just different utterances/speakers from them).

---

## 8. Robustness Testing — Recap of the "why"

Two independent stressors, tested separately:
- **Additive noise at controlled SNR** — simulates a noisy calling environment (background chatter, traffic, etc. — though white Gaussian noise is a simplification of real-world noise).
- **G.711 codec compression** — simulates the actual compression every phone call goes through, independent of noise.

Testing them separately (rather than only combined) lets me attribute *which* stressor causes *how much* degradation — that's what let me conclude noise is the bigger problem, not codec compression, which is an actionable, specific finding rather than just "performance drops in bad conditions." Phase 2 tested this on the three classical models only; Phase 3 extends it to all five models (Section 13).

---

# PHASE 3 — Interpretability, Final Evaluation & Statistical Testing

Everything below was produced by one script, `scripts/06_phase3_analysis.py` (also runnable from the Phase 3 cells of `notebooks/colab_training.ipynb`). Figures are in `outputs/phase3/plots/`, numbers in `outputs/phase3/results/`. No model was retrained in Phase 3 — every analysis uses the exact Phase 2 models, and re-running them reproduced the Phase 2 metrics to the fourth decimal place (a useful sanity check to mention).

## 9. Phase 3 at a glance

| Phase 3 plan item (from Phase 2 report) | What I did | One-line result | Key figure |
|---|---|---|---|
| SHAP for SVM, RF, Gradient Boosting | TreeSHAP (RF, GB) on 300 test clips; permutation SHAP (SVM) on 150 | RF SHAP confirms the Phase 2 ranking (ρ = 0.94); the SVM relies on *static* MFCCs instead | `shap_top20_comparison.png`, `shap_group_importance.png` |
| Gradient saliency for CNN / CNN-LSTM | Vanilla gradient, SmoothGrad, Integrated Gradients on 200 clips per model + a deletion test | Maps are faithful (deleting salient cells hurts far more than random); CNN looks at low frequencies, CNN-LSTM at the clip's start and end | `saliency_examples_*.png`, `saliency_profiles.png`, `saliency_deletion_test.png` |
| Final evaluation with significance testing | 1,000-sample bootstrap CIs; McNemar, DeLong and paired-bootstrap ΔEER tests with Holm correction | CNN-LSTM has the lowest EER but is **statistically tied with the SVM**; both beat RF and GB | `eer_confidence_intervals.png`, `significance_heatmaps.png`, `det_curves.png` |
| Noise-hardening agenda (robustness) | Noise (5–20 dB) and G.711 tested on **all five** models | Codec is fine for every model; even 20 dB noise breaks every model (EER 25–35%) | `robustness_all_models.png` |

**The four messages to land in the presentation:**
1. The handcrafted-feature story from Phase 2 survives a more rigorous test (SHAP) — delta-MFCCs and `mfcc_12_std` matter across models.
2. The deep models' saliency maps are verifiably faithful, and they reveal *where* the models look — including one region (the clip onset) that needs a follow-up check.
3. "Best model" needs statistics: CNN-LSTM vs SVM is **not** a significant difference; the clear split is {SVM, CNN, CNN-LSTM} > {RF, GB}.
4. Additive noise is the deployment blocker for *every* architecture, and the way the models fail points to a likely dataset shortcut — which defines the next step.

---

## 10. SHAP for the Classical Models

### 10.1 The idea in plain words

Shapley values come from cooperative game theory: a team wins a prize, how do you split it fairly among players? Answer: for each player, look at every possible order in which the team could have been assembled, measure how much the prize goes up at the moment that player joins, and average over all orders.

Map that onto a model:
- **"Players"** = the 257 features of one clip.
- **"Prize"** = the model's output for that clip minus the average output (the "base value").
- Each feature's **SHAP value** = its average marginal contribution.

Properties worth quoting:
- **Additivity (local accuracy):** base value + sum of all 257 SHAP values = the model's actual output for that clip. Every bit of the prediction is accounted for.
- **Consistency:** if a model changes so a feature matters more, its SHAP value never goes down (RF impurity importance doesn't guarantee this).
- **Local *and* global:** each clip gets its own explanation; averaging |SHAP| over many clips gives a global ranking.

"Removing" a feature means replacing it with values from a **background** set of typical data, so the model sees an "average" value for that feature instead.

### 10.2 How it was computed (and why two different explainers)

Exact Shapley values need every subset of features: 2²⁵⁷ subsets — impossible. Two practical approximations:

| Model | Explainer | Why | Output units | Cost |
|---|---|---|---|---|
| Random Forest | **TreeSHAP** | Exact for tree models in polynomial time — walks each tree's decision paths instead of enumerating subsets | Change in P(synthetic) | 300 clips in 36 s |
| Gradient Boosting | **TreeSHAP** | Same | Change in **log-odds** (GB's raw output) | 300 clips in < 1 s |
| SVM | **Permutation SHAP** (model-agnostic) | No tree structure to exploit; just calls the model on many masked inputs | Change in P(synthetic) | 150 clips in **25 min** |

Why the SVM is so slow: each clip needs 2×257+1 = 515 model evaluations, each against 50 background clips (~26,000 predictions per clip), and every SVM prediction computes a kernel against all 2,104 support vectors.

Details worth knowing:
- Explanations are computed on the **standardised** features (after the scaler in the pipeline), so the beeswarm colour scale is "standard deviations above/below average". SMOTE is not involved — it only runs during training.
- Explained clips are **class-balanced** samples from the test set (equal genuine and synthetic), so the ranking isn't dominated by the majority class.
- Units differ between models (probability vs log-odds), so I compare **rankings and shares**, never raw SHAP magnitudes across models.

### 10.3 How to read the plots

**Beeswarm** (`shap_beeswarm_*.png`): one row per feature (top 20), one dot per clip.
- Horizontal position = SHAP value: right of zero pushes towards **synthetic**, left pushes towards **genuine**.
- Colour = the feature's value for that clip (red = high, blue = low).
- So "red dots on the right" reads: *high values of this feature make the model say synthetic*.
- Row spread = how much influence that feature has.

**Top-20 comparison** (`shap_top20_comparison.png`): each method's importances normalised to sum to 1, plotted side by side with the Phase 2 RF impurity importance (hatched bars).

**Group importance** (`shap_group_importance.png`): importance summed by feature family.

**Dependence plot** (`shap_dependence_top3.png`): for the top 3 SVM features, feature value (x) vs SHAP value (y) for all three models — shows the *shape* of the effect (threshold-like, linear, etc.).

### 10.4 Results

**Top-5 features by mean |SHAP|:**

| SVM | Random Forest | Gradient Boosting |
|---|---|---|
| mfcc_4_mean | **delta_mfcc_6_mean** | **delta_mfcc_6_mean** |
| mfcc_6_std | **mfcc_12_std** | **mfcc_12_std** |
| **mfcc_12_std** | delta_mfcc_5_mean | mfcc_9_mean |
| mfcc_9_mean | delta_mfcc_8_mean | delta2_mfcc_0_mean |
| mfcc_2_mean | delta_mfcc_4_mean | mfcc_17_mean |

`mfcc_12_std` is in the top 3 for all three models — the single most model-independent cue.

**Cross-check against Phase 2 RF importances (the planned deliverable):**

| Comparison | Spearman ρ (all 257 features) | Top-20 overlap |
|---|---|---|
| RF SHAP vs RF importance (Phase 2) | **0.94** | **90%** |
| GB SHAP vs RF importance | 0.70 | 60% |
| SVM SHAP vs RF importance | 0.46 | 35% |
| RF SHAP vs GB SHAP | 0.69 | 60% |
| SVM SHAP vs GB SHAP | 0.47 | 55% |

**Share of importance by feature group:**

| Group (no. of features) | SVM SHAP | RF SHAP | GB SHAP | RF importance (Phase 2) |
|---|---|---|---|---|
| Static MFCC (80) | **49%** | 36% | **46%** | 35% |
| Δ-MFCC (80) | 25% | **40%** | 35% | **40%** |
| ΔΔ-MFCC (80) | 19% | 15% | 10% | 15% |
| Spectral (9) | 4% | 6% | 5% | 6% |
| Pitch (5) | 2% | 1% | 1% | 1% |
| Phase (3) | 1% | 2% | 2% | 2% |

### 10.5 What it means — how to explain it

- **The Phase 2 finding holds.** The rigorous method (SHAP) and the quick method (impurity importance) agree almost perfectly for the Random Forest (ρ = 0.94). So "delta-MFCCs dominate" was not an artefact of impurity importance's known biases.
- **Different models use different cues.** The SVM — the most accurate classical model — leans on *static* MFCC means and stds (spectral envelope shape), not deltas. Why: an RBF kernel measures distance in the full 257-dimensional space, so it uses many correlated features a little each, while trees pick a few features at each split. And static and delta MFCCs are correlated, so credit can legitimately land on either. **Takeaway line:** *"which feature is important" is a property of a model, not just of the data — so I report agreement across models rather than a single ranking.*
- **Small groups are not useless.** Spectral features are only 9 of 257 features but take 4–6% of the importance — per feature, about as influential as an average MFCC feature. Phase is 3 features at ~2% — a small but consistent signal across all methods, in line with Phase 2's rank-16 finding. Pitch matters least.

**Caveats to volunteer if asked:**
- SHAP explains the *model*, not the physics of speech — a high SHAP value means "the model relies on it", not "this is how vocoders work".
- With correlated features, credit gets split among them in ways that depend on the background set.
- Sample sizes (150–300 clips) are enough for stable rankings of the top features, not for the tail.

---

## 11. Gradient Saliency for the CNN and CNN-LSTM

### 11.1 What is being explained

For each clip I explain the model's **log-odds of "synthetic"**: logit(synthetic) − logit(genuine). Positive = the model leans synthetic. The input is the 128×128 normalised log-Mel spectrogram: 128 mel bands from 0–4,000 Hz, and 128 frames × 16 ms = **the first 2.05 s of the clip** (Section 3).

Only **correctly classified** test clips are explained (100 genuine + 100 synthetic per model) — explaining wrong predictions answers a different question.

### 11.2 The three methods

| Method | How it works | Strength | Weakness |
|---|---|---|---|
| **Vanilla gradient** | ∂(log-odds)/∂(each pixel), absolute value | One backward pass; "how sensitive is the output to this cell?" | Noisy and speckled; measures *sensitivity*, not *contribution* |
| **SmoothGrad** | Average the vanilla gradient over 16 copies of the input with small Gaussian noise added (σ = 15% of the clip's value range) | Much cleaner maps — noise averages out the speckle | Still sensitivity, not contribution |
| **Integrated Gradients (IG)** | Walk in 32 steps from a **baseline** (all zeros = a flat, average spectrum, since each spectrogram is normalised to zero mean) to the real input; average the gradients along the path; multiply by (input − baseline) | Signed (red = pushes towards synthetic, blue = towards genuine), and satisfies **completeness**: the attributions sum to log-odds(input) − log-odds(baseline) | 32× the cost; depends on the choice of baseline |

**Completeness check:** I measured how far IG's attributions were from summing to the actual output difference. The median error was **1.2% (CNN) and 1.3% (CNN-LSTM)**, so the 32-step approximation is accurate.

**Technical aside (if asked about the GPU):** cuDNN's LSTM implementation only supports backpropagation in training mode, so cuDNN is disabled while computing saliency. This lets eval-mode gradients flow through the CNN-LSTM without enabling dropout.

### 11.3 Are the maps trustworthy? The deletion test

A saliency map is just a picture unless you check it reflects what the model actually uses. The **deletion (faithfulness) test**:
1. Rank all 16,384 time-frequency cells by saliency.
2. Replace the top-k% most salient cells with the baseline value (0).
3. Re-run the model and record its probability for the *true* class.
4. Compare with deleting the same number of **random** cells.

If the map is faithful, deleting salient cells should hurt much more than deleting random cells.

**Mean true-class probability after deleting 10% of cells:**

| Model | Vanilla | SmoothGrad | Integrated Gradients | Random (control) |
|---|---|---|---|---|
| CNN | 0.62 | 0.60 | **0.54** | 0.87 |
| CNN-LSTM | 0.75 | 0.80 | **0.61** | 0.87 |

(Before deletion it is 0.995 for both.)

- Every method beats random deletion, so all three maps are faithful.
- Integrated Gradients is the most faithful of the three for both models.
- After removing just 10% of the cells IG ranks highest, the CNN's confidence in the right answer falls to 0.54 (close to a coin-flip); random deletion only gets it to 0.87.

**How to say it:** "I didn't just make heatmaps — I tested them. Deleting what the heatmaps highlight destroys the prediction; deleting random regions barely matters."

**Caveat:** deleting cells creates unnatural inputs, so part of the drop can be the model reacting to "weird" images. That is why the comparison against random deletion — which creates equally weird inputs — is the meaningful part.

### 11.4 What the models look at

**Frequency** (`saliency_profiles.png`, left; share of SmoothGrad attribution per band):

| Band | CNN genuine | CNN synthetic | CNN-LSTM genuine | CNN-LSTM synthetic |
|---|---|---|---|---|
| 0–500 Hz (28 bins = 22% of bins) | 38% | **44%** | 24% | 30% |
| 500–1000 Hz | 18% | 16% | 24% | 22% |
| 1000–2000 Hz | 21% | 19% | 26% | 24% |
| 2000–3000 Hz | 13% | 12% | 15% | 14% |
| 3000–4000 Hz | 11% | 9% | 11% | 10% |

- The **CNN** concentrates on **low frequencies (< 500 Hz)**, especially for synthetic clips. That region holds the fundamental frequency, the first harmonics and low-frequency energy — where vocoders' harmonic regularity is most visible.
- The **CNN-LSTM** spreads attention more evenly across 0–2,000 Hz (the formant region), consistent with it tracking spectral *dynamics* rather than one band.

**Time** (`saliency_profiles.png`, right):
- **CNN:** roughly uniform over time — expected, because global average pooling treats every time position equally.
- **CNN-LSTM:** strong peaks in the **first ~0.25 s** and the **last few frames**, for *both* classes.

### 11.5 The CNN-LSTM's edge focus — two explanations

This is the most interesting Phase 3 finding, so understand both halves:

1. **Architecture (a known effect).** The classifier reads only the bidirectional LSTM's **final hidden states**. The forward LSTM's final state sits at the *last* frame; the backward LSTM's final state sits at the *first* frame. Cells near the two ends of the clip have the shortest path to the output, so recurrent models naturally weight them more — the edges of the clip are privileged by construction.
2. **Data (a possible shortcut).** The first ~0.3 s of most clips is **leading silence** before the speaker starts. In the example maps, synthetic clips have *pure digital silence* there (black region, exactly zero energy), while genuine recordings have a faint microphone/room **noise floor**. A model can learn "perfectly silent start ⇒ synthetic" — a *dataset artefact*, not a property of synthetic speech. If an attacker added a little background noise, that cue disappears.

Saliency alone can't separate these two. But the robustness results (Section 13) make the shortcut explanation more likely: when noise is added — which fills the silence — the deep models start calling synthetic clips genuine.

**How to say it:** "The saliency maps flagged something a plain accuracy number would never show: the CNN-LSTM pays most attention to the clip's opening silence. That's partly how bidirectional LSTMs work, but it's also a classic dataset shortcut — and my noise results support that interpretation. The right next step is a silence-trimming ablation."

This is a strength to present, not a weakness to hide: finding problems like this is exactly what interpretability analysis is for.

---

## 12. Final Evaluation and Statistical Testing

### 12.1 Why statistics are needed

The test set has 2,734 clips. On the standard test set the CNN-LSTM makes **15 errors** (2 false alarms + 13 misses) and the SVM makes **24** (4 + 20). A difference of nine clips could easily come from *which* clips happened to land in the test set. Significance testing answers: *"if I drew a different test set, would the ranking hold?"*

### 12.2 Bootstrap confidence intervals

- **Method:** resample the test set **with replacement** 1,000 times (stratified — each resample keeps the same genuine/synthetic counts), recompute EER/AUC/F1 each time, and take the 2.5th and 97.5th percentiles as the **95% confidence interval**.
- **Why bootstrap:** EER has no simple formula for its variance; the bootstrap needs no distributional assumptions.

| Model | EER, standard test (95% CI) | EER, Edge-TTS generalisation (95% CI) | AUC, standard test (95% CI) |
|---|---|---|---|
| SVM | 0.83% (0.35–1.31) | 0.67% (0.13–1.73) | 0.9992 (0.9982–0.9998) |
| Random Forest | 3.05% (2.27–3.88) | 2.93% (1.87–3.82) | 0.9956 (0.9935–0.9974) |
| Gradient Boosting | 3.46% (2.45–4.57) | 3.11% (2.04–4.76) | 0.9949 (0.9927–0.9967) |
| CNN | 0.81% (0.42–1.20) | 0.36% (0.09–0.98) | 0.9993 (0.9984–0.9998) |
| **CNN-LSTM** | **0.35% (0.10–0.76)** | **0.04% (0.00–0.27)** | **0.9996 (0.9991–0.9999)** |

**Reading it:** the intervals for SVM, CNN and CNN-LSTM **overlap heavily**; RF and GB sit clearly higher. The Edge-TTS intervals are wider because that set is smaller (1,505 clips, only 372 genuine).

### 12.3 The three pairwise tests

All 10 model pairs were tested on each test set (`significance_heatmaps.png`):

**McNemar's test — "do they make different mistakes?"**
- Uses the *hard decisions* at each model's deployed threshold. Two models are tested on the same clips, so only the clips where they **disagree** carry information: *b* = clips A got right and B got wrong; *c* = the reverse.
- If the models were equally good, *b* ≈ *c*. Test statistic: χ² = (|b − c| − 1)² / (b + c), with continuity correction; an exact binomial test is used when b + c < 25.
- Example: CNN vs CNN-LSTM on the standard test: b = 9, c = 44 → p < 0.0001.
- This was the test named in the project config (`significance_test: mcnemar`).

**DeLong's test — "is one AUC really higher?"**
- Compares two ROC AUCs measured on the *same* clips, accounting for the correlation between them (both models find the same clips easy or hard).
- Threshold-free: it tests *ranking* quality, not decisions.

**Paired bootstrap on EER — "is one EER really lower?"**
- Uses the same 1,000 resamples for every model, so each resample gives a paired difference ΔEER = EER(A) − EER(B).
- 95% CI of ΔEER; if it includes 0, the difference isn't significant.
- Tests the project's primary metric directly.

**Holm–Bonferroni correction:**
- Running 10 tests at p < 0.05 gives roughly a 40% chance of at least one false "significant" result.
- Holm sorts the p-values, multiplies the smallest by 10, the next by 9, and so on (never letting an adjusted value fall below the previous one).
- It is less conservative than plain Bonferroni (multiply all by 10) but still controls the family-wise error. All reported p-values are Holm-adjusted.

### 12.4 Results — what is and isn't significant (standard test set)

| Pair | McNemar | DeLong | ΔEER bootstrap | ΔEER (A − B), 95% CI | Verdict |
|---|---|---|---|---|---|
| SVM vs CNN-LSTM | 0.38 | 1.00 | 0.76 | +0.39 pp (−0.15, +0.99) | **Not significant — tied** |
| SVM vs CNN | 0.008 | 1.00 | 1.00 | +0.02 pp (−0.59, +0.61) | Same EER/AUC; different error patterns |
| CNN vs CNN-LSTM | **< 0.0001** | 1.00 | 0.76 | +0.37 pp (−0.15, +0.89) | Same ranking quality; CNN-LSTM makes fewer *decision* errors |
| SVM vs RF / GB | < 0.0001 | ≤ 0.0001 | < 0.0001 | −2.3 / −2.6 pp | **SVM significantly better** |
| CNN-LSTM vs RF / GB | < 0.0001 | ≤ 0.0007 | < 0.0001 | −2.7 / −3.0 pp | **CNN-LSTM significantly better** |
| RF vs GB | 0.51 | 1.00 | 1.00 | −0.32 pp | Tied |

(p-values Holm-adjusted; pp = percentage points.) Not every cell is significant on every test — e.g. CNN vs RF on McNemar (p = 0.13). The one test that separates the two groups everywhere is the **paired-bootstrap EER test** (the primary metric).

The Edge-TTS results tell the same story with less power (smaller set): {SVM, CNN, CNN-LSTM} are mutually tied after correction, and all three beat RF and GB on EER (p ≤ 0.01). Some AUC comparisons there (e.g. SVM vs GB, CNN vs RF) are not significant.

### 12.5 Why do the tests disagree for CNN vs CNN-LSTM?

This is a likely examiner question, and the answer shows understanding:
- McNemar looks at **decisions at the deployed threshold**. The CNN uses a 0.5 probability threshold and misses **48** synthetic clips there, against the CNN-LSTM's 13.
- DeLong and the EER test look at **ranking quality** (threshold-free), and there the two are indistinguishable.
- So the CNN's problem is mostly **a poorly placed threshold (calibration)**, not a worse ability to separate the classes. Moving the CNN to its EER threshold would close most of the gap.
- Note the protocol: classical models use the validation-set EER threshold (from Phase 2); deep models use argmax (P ≥ 0.5), as in Phase 2.

### 12.6 The defensible headline

> "The CNN-LSTM achieves the lowest EER on both test sets (0.35% and 0.04%), but it is **statistically indistinguishable from the SVM** on EER and AUC. SVM, CNN and CNN-LSTM all significantly outperform Random Forest and Gradient Boosting on EER, on both test sets."

Practical implication: the SVM — far cheaper, with no GPU and interpretable inputs — is a legitimate production choice. This strengthens the Phase 2 suggestion of an SVM screening stage with the CNN-LSTM as a second opinion.

### 12.7 DET curves (`det_curves.png`)

- A **DET (Detection Error Trade-off)** curve plots the miss rate against the false-acceptance rate, both on a **normal-deviate (probit) scale**.
- It is the standard anti-spoofing plot (ASVspoof uses it) because at error rates near 1% ROC curves all hug the corner and look identical, while the probit scale spreads them out.
- Where a curve crosses the diagonal is its EER; lower-left is better.

---

## 13. Robustness of All Five Models

### 13.1 Setup

- 300 test clips, class-balanced (150 genuine + 150 synthetic), drawn from the raw audio.
- **Seven conditions:** clean; white noise at 20, 15, 10 and 5 dB SNR; G.711 alone; and G.711 + 10 dB noise. Noise is added *before* the codec, as on a real call: background noise at the caller's end, then the phone network compresses it.
- **Paired design:** each clip gets exactly the same noise realisation for every model (seeded random generator), so differences between models come from the models, not from luck.
- Features and spectrograms were re-extracted from the degraded audio exactly as in training.

Besides threshold-free EER, I also report the error rates **at the deployed threshold**, because that is what a bank would actually experience:
- **FAR** = genuine callers wrongly flagged as synthetic (customer friction).
- **FRR** = synthetic speech wrongly let through (the security failure).

(These match this script's column names; some papers swap the two terms, so define them when you say them.)

### 13.2 Results — EER (%)

| Model | Clean | 20 dB | 15 dB | 10 dB | 5 dB | G.711 | G.711 + 10 dB |
|---|---|---|---|---|---|---|---|
| SVM | 0.3 | 32.0 | 38.3 | 44.0 | 48.7 | 3.0 | 40.7 |
| Random Forest | 2.0 | 32.3 | 32.7 | 40.0 | 38.0 | 5.3 | 32.7 |
| Gradient Boosting | 4.0 | 30.0 | 32.0 | 34.7 | 34.7 | 8.3 | 27.7 |
| CNN | 1.3 | 35.3 | 41.7 | 41.0 | 44.7 | 1.3 | 44.0 |
| **CNN-LSTM** | **0.3** | **24.7** | 32.7 | 35.3 | 40.3 | **1.0** | 31.0 |

(50% EER = coin-flip.) The SVM figures agree with Phase 2 (≈ 32% at 20 dB, ≈ 45–49% at 5 dB) — a good consistency check.

### 13.3 What it means

1. **The codec is not a problem — especially for the deep models.** Under G.711, CNN-LSTM EER is 1.0% and CNN 1.3%, versus 3–8% for the classical models. The spectrogram models handle telephone compression best.
2. **Noise breaks every model, deep ones included.** Even mild 20 dB noise (a fairly clean call) pushes EER to 25–35%. The CNN-LSTM degrades least at 20 dB (24.7%), but nothing is deployable under noise. Phase 2 showed this for classical models only; Phase 3 shows deep learning does **not** solve it.
3. **The models fail in opposite directions** (`robustness_all_models.png`, middle and right panels):
   - **CNN, CNN-LSTM, Random Forest:** with noise they call almost everything **genuine** — about **90% of synthetic clips get through** (FRR ≈ 0.9), while genuine callers are almost never flagged. *This is the dangerous failure for a bank: silent acceptance of attacks.*
   - **SVM:** the reverse — at 5 dB it flags **99% of genuine callers**. Safe but unusable.
   - **Gradient Boosting:** the most balanced degradation (lowest EER at 10 dB and with G.711 + 10 dB noise), consistent with Phase 2 calling it the most noise-robust classical model.
4. **Threshold re-tuning alone won't fix it.** At 20 dB, AUC is still 0.70–0.82, so some ranking ability remains, but EER of 25–35% means no threshold gives acceptable errors. The models themselves must change.

### 13.4 Connecting the dots: the shortcut hypothesis

Put Sections 11.5 and 13.3 together:
- The CNN-LSTM focuses on the clip onset, where synthetic clips have digital silence and genuine ones have a noise floor.
- Adding noise *fills that silence*, making synthetic clips look like genuine recordings.
- And that is exactly the observed failure: under noise, the deep models (and RF) **call synthetic clips genuine**.

So a substantial part of the clean-condition performance may come from a **recording-condition cue**, not just a speech-synthesis cue. This is **consistent with, not proof of**, the hypothesis.

**How to test it (future work — say this confidently):**
1. **Silence-trimming ablation:** trim leading and trailing silence with voice-activity detection, re-extract spectrograms, and re-evaluate. If EER rises sharply, the shortcut is confirmed.
2. **Noise-augmented training:** train with noise at 5–20 dB SNR (the config already lists these levels under `augmentation`). This removes the cue and forces the model to learn speech-based artefacts.
3. **Realistic noise:** babble and street noise, not just white noise.
4. **Pre-trained front-ends** (wav2vec 2.0 — the carried-forward stretch goal), which are known to be more robust.

Wider context: the ASVspoof community has reported the same issue — silence duration in ASVspoof 2019 LA is itself predictive of spoofing (Müller et al., 2021, *"Speech is Silver, Silence is Golden"*). Citing it shows the finding fits known literature rather than being a bug.

---

## 14. Likely Questions on Phase 3 — and Answers

**Q: Why use SHAP when you already had Random Forest importances?**
A: Impurity importance is model-specific, biased towards continuous features, and only gives a global ranking. SHAP is model-agnostic, works for the SVM too, gives per-clip explanations, and has a theoretical fairness guarantee. And it confirmed the Phase 2 ranking (ρ = 0.94), so the Phase 2 conclusion now rests on a stronger method.

**Q: Why does the SVM rank features differently from the trees?**
A: An RBF kernel uses distances in the full feature space, spreading reliance across many correlated features; trees pick a few features per split. Static and delta MFCCs are correlated, so credit can move between them. Feature importance is a property of the model, which is why I report agreement across models.

**Q: Why not use SHAP on the CNNs too?**
A: The CNNs' input is 16,384 spectrogram cells — permutation-style SHAP would be extremely expensive, and SHAP's gradient-based variants are essentially Integrated Gradients, which I used. IG is the deep-learning counterpart of SHAP; both satisfy completeness.

**Q: How do you know the saliency maps aren't just pretty noise?**
A: The deletion test. Removing the top 10% IG-ranked cells drops true-class confidence to 0.54 (CNN) and 0.61 (CNN-LSTM); removing random cells only gets it to 0.87. IG's completeness error was about 1%.

**Q: Your best model isn't significantly better than the SVM — isn't that disappointing?**
A: It's an honest and useful result. At sub-1% EER on about 2,700 clips, the differences are a handful of clips. It means a cheap, interpretable SVM is a valid deployment option, and I can claim a clear, significant gap over RF and GB.

**Q: Why three different significance tests?**
A: They answer different questions. McNemar tests decisions at the operating threshold (what a deployment experiences); DeLong tests ranking ability via AUC; the paired bootstrap tests the primary metric, EER. Where they disagree — CNN vs CNN-LSTM — it tells us something specific: the CNN's threshold is poorly calibrated.

**Q: Why Holm correction?**
A: Ten comparisons per test set inflate false positives; Holm controls the family-wise error rate while being less conservative than Bonferroni.

**Q: Is the model detecting synthetic speech or detecting silence?**
A: Possibly partly silence — and I found that through my own interpretability analysis, which is the point of doing it. The saliency onset focus plus the "noise makes synthetic look genuine" failure are consistent with a silence shortcut, which is also documented for ASVspoof 2019. The fix and the test are clear: silence trimming and noise-augmented training.

**Q: Why does noise hurt so much more than the codec?**
A: G.711 is a deterministic quantisation that preserves the spectral envelope and timing, so the cues survive. Additive noise fills the spectrogram's quiet regions and masks fine harmonic and phase structure — exactly where the models' cues live, including the silence cue.

**Q: Why only 300 clips for robustness?**
A: Handcrafted feature extraction (especially pitch tracking with pYIN) is slow — 300 clips × 7 conditions took about 15 minutes on 8 CPU cores. The clips are class-balanced, and each one gets identical noise for every model, so the model comparison is paired and fair. The degradation is so large (0.3% → 25–49%) that sample size doesn't change the conclusion.

---

## 15. Quick-fire glossary (for on-the-spot recall)

| Term | One-line definition |
|---|---|
| Mel scale | Frequency scale approximating human pitch perception (non-linear, denser at low frequencies) |
| FFT | Fast Fourier Transform — converts a time-domain signal into its frequency-domain spectrum |
| DCT | Discrete Cosine Transform — decorrelates/compresses Mel-log-energies into MFCCs |
| Vocoder | The component of a TTS system that converts a spectrogram/acoustic representation into a raw waveform |
| Logical Access (ASVspoof) | Attack type: synthetic/converted speech (as opposed to "Physical Access" = replay attacks) |
| Bidirectional LSTM | LSTM run both forward and backward over a sequence, outputs concatenated |
| Epoch | One full pass through the entire training dataset |
| Batch size | Number of examples processed together before one gradient update |
| Overfitting | Model memorizes training data patterns that don't generalize to new data |
| Regularization | Any technique (dropout, weight decay, early stopping, etc.) that discourages overfitting |
| Class imbalance | Unequal number of examples per class (here, ~1:3.7 genuine:synthetic) |
| Checkpoint | A saved snapshot of model weights at a given point in training |
| Shapley value | Fair share of a prediction credited to one feature, averaged over all orders of adding features |
| SHAP base value | The model's average output; base value + sum of SHAP values = this clip's output |
| TreeSHAP | Exact, fast SHAP algorithm for tree ensembles (RF, GB) |
| Permutation SHAP | Model-agnostic SHAP via many masked model calls (used for the SVM; slow) |
| Background set | Typical data used to "switch off" a feature when computing SHAP |
| Spearman ρ | Rank correlation — do two methods order the features the same way? (1 = identical order) |
| Saliency map | Heatmap of how much each input cell affects the model's output |
| SmoothGrad | Gradient saliency averaged over noisy copies of the input, to reduce speckle |
| Integrated Gradients | Gradients averaged along a path from a baseline to the input; attributions sum to the output change |
| Completeness | IG property: attributions add up exactly to output(input) − output(baseline) |
| Deletion test | Remove the most-salient cells and check the prediction drops more than with random removal |
| Shortcut learning | Model relies on a spurious cue (e.g. leading silence) instead of the intended signal |
| Bootstrap CI | Confidence interval from recomputing a metric on many resampled test sets |
| McNemar's test | Paired test on the clips where two classifiers disagree |
| DeLong's test | Test for the difference between two correlated ROC AUCs |
| Holm–Bonferroni | Step-down correction for multiple comparisons (controls family-wise error) |
| DET curve | Miss rate vs false-acceptance rate on a probit scale — the standard anti-spoofing plot |
| FAR / FRR (as used here) | Genuine flagged as synthetic / synthetic passed as genuine, at the deployed threshold |
