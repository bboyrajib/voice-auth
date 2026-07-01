# Speaker Script — Midterm Evaluation (Phase 1 + Phase 2)
**Detection of AI-Generated and Cloned Speech for Banking Voice Authentication**
Companion to `Phase2_Midterm_Presentation.pptx` (17 slides). Aim for ~15–18 minutes total, leaving room for questions.

---

### Slide 1 — Title
"Good [morning/afternoon]. My project is on detecting AI-generated and cloned speech to strengthen voice-based authentication in banking environments. This is my midterm evaluation, covering everything completed in Phase 1 and Phase 2. My internal guide is Dr. Anil Rahate from Wipro, and my IISc faculty mentor is Prof. Chandra Sekhar Seelamantula."

### Slide 2 — Agenda
"Quickly, here's how I'll structure this: I'll start with the problem and research questions, then a one-slide timeline showing where we are. I'll recap Phase 1 briefly since it's the foundation, then spend most of the time on Phase 2 — the classical baselines, feature importance, the two deep learning models, and robustness testing. I'll close with consolidated results, key insights, and the Phase 3 plan."

### Slide 3 — The Problem
"Voice authentication is now routine in banking — IVR systems, call centres, transaction confirmations by voice. The issue is that neural text-to-speech and voice cloning have advanced to the point where a few seconds of someone's voice is enough to synthesize convincing speech in that voice. Existing voice biometric systems were built to catch replay attacks — playing back a recording — not this generation of neural TTS. So there's a real gap: if an attacker clones a customer's voice, many current systems won't catch it. My project's goal is to build and validate a detector that specifically works under banking telephony conditions — short utterances, compressed audio, background noise — not just clean lab recordings."

### Slide 4 — Research Questions
"I structured the project around four research questions. First, can supervised ML reliably tell real speech from synthetic speech, even on speakers and TTS systems it hasn't seen before? Second, what acoustic properties actually give away synthetic speech — is it spectral shape, phase artifacts, prosody? Third, how do these models hold up specifically under banking channel conditions — codec compression, noise, short phrases? And fourth, do deep learning models actually outperform classical ML here, or is that not obviously true? Phase 2 is where I get concrete answers to all four."

### Slide 5 — Project Timeline
"Quick orientation: Phase 1, January to April, was literature review, dataset construction, and the feature pipeline — that's complete. Phase 2, May to July, is model training and evaluation — baselines, deep learning, robustness — also complete, and that's the bulk of today's talk. Phase 3, starting in August, is interpretability, final statistical testing, and thesis writing. So we're right at the Phase 2 to Phase 3 boundary."

### Slide 6 — Dataset (Phase 1 recap)
"Just a quick recap since this is the foundation everything else sits on. I built a dataset of just under 18,000 utterances from 10 different sources — genuine speech from LibriSpeech, and synthetic speech from ASVspoof 2019's six TTS and voice-conversion systems, plus VITS, Tacotron2, and Microsoft Edge-TTS. Everything is processed at 8 kHz with G.711 codec simulation to mimic an actual phone call, and trimmed to 3–8 seconds — about the length of a transaction confirmation phrase. Critically, I held out Edge-TTS entirely — zero Edge-TTS clips in training — so I have a genuine test of generalization to a TTS system the model has never seen."

### Slide 7 — Feature Pipeline (Phase 1 recap)
"For features, I built two parallel representations. A 257-dimensional handcrafted vector — MFCCs with delta and delta-delta coefficients, spectral centroid, bandwidth, rolloff, zero-crossing rate, pitch, harmonic-to-noise ratio, and phase difference statistics — for the classical models. And 128-bin log-Mel spectrograms for the deep learning models. Everything is config-driven and validated by an automated diagnostic script before any training happens."

### Slide 8 — Classical ML Baselines
"Now into Phase 2 proper. I trained three classical models on the handcrafted features: SVM with an RBF kernel, Random Forest, and Gradient Boosting, all tuned with grid search under 5-fold cross-validation, with SMOTE to handle class imbalance. The standout result is the SVM: an Equal Error Rate of 0.83% on the standard test set, and it actually does *better* — 0.67% EER — on the fully unseen Edge-TTS set. All three models generalize well to Edge-TTS, which is an early signal that these features are capturing something general about synthetic speech, not just quirks of the training TTS systems."
*(Point at the table — no need to read every number, just call out the SVM row and the "gen. test" column.)*

### Slide 9 — Feature Importance
"To understand *why* these models work, I ran a Random Forest feature importance analysis. The clear winners are delta-MFCC coefficients — the ones capturing how the spectral envelope changes moment to moment. The intuition is that TTS vocoders produce smoother, more mechanically regular spectral transitions than real human articulation, and these delta features pick that up directly. Phase-difference features also show up around rank 16, which validates including phase-based features — TTS vocoders introduce phase coherence artifacts that magnitude-only features like plain MFCCs would miss."

### Slide 10 — Deep Learning Architectures
"For deep learning, I built two models on the log-Mel spectrograms. A CNN — four convolutional blocks with batch norm and max-pooling, about 420,000 parameters. And a CNN-LSTM, which uses the same convolutional front-end but pools only across frequency, keeping the time axis intact, then feeds that sequence into a bidirectional LSTM. That's about 1.8 million parameters. The design intent here is direct: the CNN alone can only learn static spectral patterns, while the CNN-LSTM can explicitly model how those patterns evolve over time — which matters because prosody and vocoder smoothing are inherently temporal phenomena."

### Slide 11 — Deep Learning Results
"Training-wise, both converged within about 20–22 epochs on my RTX 3060, roughly an hour each. But the gap in the results is the interesting part: CNN validation EER of 1.5%, versus CNN-LSTM at 0.9%, and that gap holds on the test set too — the CNN-LSTM clearly benefits from that explicit temporal modeling. The ROC curves on screen show all five models — three classical plus two deep learning — and you can see the CNN-LSTM curve is essentially hugging the top-left corner, with an AUC of 1.000 on the generalization test."

### Slide 12 — Confusion Matrices
"Just to ground this in concrete numbers rather than just rates: on the standard test set of 2,734 clips, the CNN and CNN-LSTM each produce fewer than 3 false positives and under 50 false negatives. In a banking context, false positives — flagging a genuine customer as synthetic — are usually more costly than false negatives, so that's an encouraging number."

### Slide 13 — Robustness Evaluation
"This is where I stress-tested the models against real banking channel conditions. Two variables: additive noise at different SNR levels, and G.711 codec compression. The codec compression alone is fairly benign — EER only rises to 5–8% across models. Noise is the real problem: SVM's EER climbs to about 45% at 5 dB SNR, which is essentially unusable. Gradient Boosting is the most noise-robust of the classical models, flattening around 33%. The takeaway I want to highlight: for deployment, noise-handling — not codec robustness — should be the priority, whether that's upstream noise suppression or training on noise-augmented data, which is exactly what I've scoped into Phase 3."

### Slide 14 — Consolidated Results
"This table is the full picture — all five models, both the standard test set and the Edge-TTS generalization test. Two things to take away: one, CNN-LSTM is the best model overall, with an EER of 0.35% on test and 0.04% on the unseen Edge-TTS set. Two — and this is the finding I find most interesting — *every single model* does as well or better on the fully unseen TTS system as it does on the system it was trained on. That's not what I expected going in."

### Slide 15 — Key Insights
"Three insights I'd highlight from Phase 2. First, the deep learning advantage here is specifically about temporal modeling — the LSTM stage — not convolution in general; a well-tuned SVM on handcrafted features is still very competitive, even beating the plain CNN. Second, generalization to an unseen TTS system was stronger than I expected, suggesting these models are picking up on properties common to neural TTS architectures broadly, not fingerprints of the specific systems in training. Third, noise is a bigger deployment threat than codec compression, which reorders my priorities for hardening this for actual banking use."

### Slide 16 — Phase 3 Plan
"Looking ahead: Phase 3 starts with interpretability work I haven't done yet — SHAP analysis for the classical models, and gradient saliency mapping for the CNN and CNN-LSTM, to visualize exactly which time-frequency regions drive their decisions. After that, final evaluation with formal statistical significance testing between models, and then thesis writing and submission through October–November."

### Slide 17 — Thank You
"That's Phase 1 and Phase 2. Happy to take questions, or go deeper into any of the results, the architecture choices, or the robustness numbers."

---

## Anticipated questions (quick answers to have ready)

**Q: Why does every model generalize *better* to Edge-TTS than to the standard test set?**
"My working explanation is that Edge-TTS, as a single commercial system, produces relatively consistent artifacts, whereas the standard test set mixes six-plus different ASVspoof TTS/VC systems with more varied artifact profiles — so the standard test set is actually the harder, more diverse evaluation. It's not that the model is over-fit to Edge-TTS; it never saw it during training."

**Q: Why is SVM competitive with the CNN?**
"The handcrafted features — especially delta-MFCCs and phase difference — already encode a lot of the same discriminative signal the CNN has to learn implicitly from raw spectrograms. The CNN's advantage only shows up once you add explicit temporal modeling via the LSTM."

**Q: What's your primary metric and why?**
"Equal Error Rate — it's the standard metric in anti-spoofing / voice biometrics literature because it balances false acceptance and false rejection at a single, comparable operating point, rather than being threshold-dependent like raw accuracy."

**Q: What's not done yet?**
"SHAP-based interpretability for the classical models, gradient saliency maps for the deep models, and formal statistical significance testing between models — all scoped into Phase 3, starting next month."

**Q: Any deviations from the approved plan?**
"None on the required deliverables. The one optional stretch goal — fine-tuning a frozen wav2vec 2.0 encoder — wasn't attempted since the CNN-LSTM already performs near-ceiling on available compute; that's noted as an optional Phase 3 item, not a deviation."
