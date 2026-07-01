# Technical Crash Course — Everything in the Pipeline, Explained

Purpose: so you can field any technical question about the pipeline confidently, even if it's not directly covered in the reports. Written assuming you know ML/DL basics (what training/testing is, what a neural net is) but want the *specifics* of every technique used here.

---

## 1. Audio & Signal Processing Basics

**Sampling rate (8 kHz).** Audio is a continuous wave; a computer stores it as discrete samples per second. 8 kHz means 8,000 samples/second. This is the standard rate for telephone-quality audio (landline/mobile voice calls use 8 kHz — it's enough to capture human speech intelligibly, roughly up to 4 kHz of frequency content, per the Nyquist theorem: max frequency captured = half the sampling rate). Music/studio recordings use 44.1–48 kHz. I deliberately downsample everything to 8 kHz because that's what a real banking call actually sounds like — training on studio-quality 44 kHz audio and testing on 8 kHz phone audio would be an unrealistic mismatch.

**G.711 codec.** This is the actual audio compression codec used on the global telephone network (both landline and most VoIP). It's a "µ-law" (mu-law) companding scheme — it compresses the dynamic range of the signal non-linearly (more resolution for quiet sounds, less for loud ones, mimicking human hearing sensitivity) to fit into 8 bits/sample instead of 16. I encode and then decode through G.711 to introduce the exact quantization artifacts a real phone call would have. Why this matters for the project: TTS systems are usually evaluated on clean, uncompressed audio, but a fraud attack over a bank's phone line would go through this codec — so my evaluation reflects the real threat model.

**Why 3–8 second clips?** That's the typical length of a spoken transaction-confirmation phrase ("Yes, I confirm this transfer of five thousand rupees") — not a long conversation. Longer utterances give models more signal to work with, so testing on realistically short clips is a harder, more honest test.

**SNR (Signal-to-Noise Ratio), measured in dB.** Ratio of the power of the actual speech signal to the power of background noise, in decibels (logarithmic scale). Higher dB = cleaner audio. 20 dB SNR is a fairly clean call; 5 dB SNR is quite noisy (think: someone on a call from a crowded street). I added white Gaussian noise at 5/10/15/20 dB to simulate this. "White Gaussian noise" = random noise with equal energy at every frequency, statistically Gaussian-distributed amplitude — a standard, tractable noise model, though not a perfect stand-in for real background chatter (a limitation I'd acknowledge if asked).

---

## 2. Handcrafted Features (the 257-dimensional vector)

These are numbers computed directly from the audio waveform using signal-processing formulas — no learning involved, just math. Extracted with a 25 ms analysis window sliding forward 10 ms at a time (so windows overlap by 15 ms) — this is the standard "frame" size in speech processing, roughly matching how fast the vocal tract physically changes shape.

**MFCC (Mel-Frequency Cepstral Coefficients) — 40 coefficients × {static, Δ, ΔΔ} = 120 dims.**
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

Instead of collapsing the spectrum into a handful of numbers (like MFCCs), for the deep learning models I feed in something closer to a raw picture: a **128-bin log-Mel spectrogram** — the Mel-scaled spectrum (128 frequency bands) computed at every 10 ms time-step, log-compressed (because energy differences are perceived logarithmically, and it also compresses the numeric range), and normalized. This creates a 2D image: one axis is time, one is frequency (Mel-scaled), and pixel intensity is log-energy. It's resized/cropped/padded to a fixed 128×128 grid so it can be fed into a CNN like an image.

**Why not just use MFCCs for deep learning too?** MFCCs already threw away a lot of detail via the DCT compression step — that's fine for a classical ML model with a fixed, engineered feature set, but a CNN can learn its *own* useful representations directly from the richer, less-compressed Mel spectrogram, which is why deep learning models conventionally use spectrograms rather than MFCCs as input.

---

## 4. Classical ML Models

**SVM (Support Vector Machine) with RBF kernel.**
- Core idea: find the decision boundary (hyperplane) that maximizes the margin (distance) between the two classes.
- **RBF (Radial Basis Function) kernel:** SVMs can only draw a straight-line boundary in the original feature space; a kernel implicitly maps the data into a higher-dimensional space where a straight boundary *can* separate classes that aren't linearly separable in the original space. RBF is essentially a similarity measure based on distance — points close together in feature space get high similarity, far apart get low similarity, decaying like a Gaussian.
- **C = 10 (regularization parameter):** controls the trade-off between a wide margin (simpler, more generalizable boundary) and correctly classifying every training point (risk of overfitting). Higher C = less tolerance for misclassified training points, tighter fit. C=10 was found via grid search to be a good balance.
- **γ (gamma) = "auto":** controls how far the influence of a single training point reaches in the RBF kernel. Small gamma = smooth, far-reaching influence (simpler boundary); large gamma = each point only influences its immediate neighborhood (can overfit). "auto" sets it based on the number of features (1/n_features), a reasonable default that the grid search confirmed worked well.

**Random Forest.**
- An ensemble of many decision trees (300 in my setup), each trained on a random subset of the data (bootstrap sampling) and a random subset of features at each split. Final prediction = majority vote across all trees.
- Why it works: individual decision trees overfit easily; averaging many "weakly correlated" trees cancels out their individual errors while keeping their collective signal — this is the general "bagging" (bootstrap aggregating) principle.
- **Unlimited depth:** each tree is allowed to grow until leaves are pure (or some minimum sample count) — depth isn't artificially capped, since the ensemble averaging already controls overfitting.
- **Feature importance:** Random Forest gives you a free-standing importance score per feature, computed from how much each feature reduces impurity (Gini impurity — a measure of class mixing) across all the splits that use it, averaged across all trees. This is what produced the delta-MFCC ranking. (Caveat if asked: this is *not* SHAP — it's a simpler, model-specific importance measure, and it can be biased toward high-cardinality/continuous features. SHAP is the more rigorous, model-agnostic version, planned for Phase 3.)

**Gradient Boosting.**
- Also an ensemble of decision trees (200, in my setup), but built *sequentially* rather than independently: each new tree is trained specifically to correct the errors (residuals) of the ensemble so far, weighted by a **learning rate (0.1)** that controls how much each new tree is allowed to contribute (small steps = more stable, needs more trees).
- Contrast with Random Forest: Random Forest reduces variance by averaging independent trees; Gradient Boosting reduces bias by iteratively focusing on mistakes. In my results, Gradient Boosting turned out to be the most noise-robust classical model — likely because its sequential error-correction process builds a more calibrated decision boundary near the noisy/ambiguous region, though Random Forest and SVM had lower clean-condition EER.

**SMOTE (Synthetic Minority Oversampling Technique).**
- My dataset is imbalanced — roughly 1 genuine : 7 synthetic clips. If you train naively on imbalanced data, the model can get high accuracy by just always predicting the majority class.
- SMOTE generates *synthetic* new minority-class (genuine) examples by interpolating between existing minority examples and their nearest neighbors in feature space — not just duplicating them, which would just overweight existing points without adding new information.
- `class_weight='balanced'` is a complementary approach used simultaneously: it reweights the loss function so mistakes on the minority class count more, without changing the actual data.

**Stratified 5-fold cross-validation + grid search.**
- Grid search = systematically trying combinations of hyperparameters (like C and gamma for SVM) and picking the best-performing combination.
- 5-fold CV = split the training data into 5 chunks; train on 4, validate on the 5th, rotate through all 5 combinations, average the validation score. This gives a more reliable performance estimate than a single train/validation split, since every data point gets used for both training and validation across the 5 rounds.
- "Stratified" = each fold preserves the same class ratio as the full dataset — important here given the 1:7 imbalance, so no fold accidentally ends up with too few genuine examples.
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
- **Dropout (p=0.5):** during training, randomly zero out 50% of neurons in a layer for each batch. Prevents the network from over-relying on any specific neuron/pathway, a standard regularization technique against overfitting.
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
- **Weighted cross-entropy loss:** cross-entropy is the standard classification loss (penalizes confident-wrong predictions heavily via a log term); "weighted" means the minority class's loss contributes more, addressing the 1:7 imbalance the same way `class_weight='balanced'` does for the classical models.
- **WeightedRandomSampler:** rather than (or in addition to) reweighting the loss, this controls *which examples get sampled into each mini-batch* — oversampling the minority class so batches are roughly balanced by construction, giving the model more frequent exposure to genuine examples.
- **Mixed precision (float16 autocast):** most computation is done in 16-bit floating point instead of 32-bit, which is faster and uses less GPU memory, with some operations kept in 32-bit for numerical stability. This is what made training feasible on a 6.4 GB laptop GPU.
- **Early stopping (patience 10):** stop training if validation EER hasn't improved for 10 consecutive epochs, and keep the best checkpoint — protects against overfitting from training too long.

---

## 6. Evaluation Metrics

**Accuracy.** Fraction of correct predictions. Misleading under class imbalance (predicting "synthetic" for everything gets ~87.5% accuracy on a 1:7 dataset without learning anything).

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

Testing them separately (rather than only combined) lets me attribute *which* stressor causes *how much* degradation — that's what let me conclude noise is the bigger problem, not codec compression, which is an actionable, specific finding rather than just "performance drops in bad conditions."

---

## 9. Interpretability (what's coming in Phase 3 — know the distinction if asked)

**Random Forest feature importance (done, Phase 2).** Model-specific, based on impurity reduction. Fast to compute, but can be biased and doesn't explain *individual* predictions — just an overall ranking.

**SHAP (SHapley Additive exPlanations) — planned, Phase 3.** Based on Shapley values from cooperative game theory: for each prediction, treat each feature as a "player" contributing to the outcome, and fairly distribute credit for the prediction across all features by averaging over all possible orderings in which features could be "added" to the model. Model-agnostic (works on any model) and gives both global importance *and* per-prediction explanations ("this specific clip was flagged as synthetic mainly because of X, Y, Z"). More rigorous and more expensive to compute than RF's built-in importances — this is why I'm treating them as genuinely different things and not conflating "RF importance done" with "SHAP done."

**Gradient saliency maps — planned, Phase 3 (for CNN/CNN-LSTM).** Compute the gradient of the model's output (predicted class score) with respect to each input pixel (each time-frequency bin of the spectrogram). Large gradient magnitude at a pixel means small changes there would strongly affect the prediction — i.e., the model is "paying attention" to that region. Visualized as a heatmap overlaid on the spectrogram. This is the deep-learning analogue of feature importance, but spatially localized within a single spectrogram rather than a global ranking across the 257 handcrafted features.

---

## 10. Quick-fire glossary (for on-the-spot recall)

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
| Class imbalance | Unequal number of examples per class (here, ~1:7 genuine:synthetic) |
| Checkpoint | A saved snapshot of model weights at a given point in training |
