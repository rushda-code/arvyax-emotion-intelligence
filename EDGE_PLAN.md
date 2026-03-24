# Edge and Offline Deployment Plan — ArvyaX
### Running the Emotional Intelligence System On-Device

---

## Overview

ArvyaX is a wellness product. The emotional understanding system must work:
- Without internet connectivity (forest retreats, meditation spaces)
- On mobile hardware (smart yoga mat companion app)
- With low latency (real-time feedback during or after a session)
- Without exposing private journal entries to external servers

This document outlines how to deploy the system on mobile and edge devices.

---

## Current System — Size and Speed

Component              Size          Latency (mobile CPU estimate)
TF-IDF vectorizer      ~500KB        less than 5ms
XGBoost classifier     ~2-5MB        less than 20ms
XGBoost regressor      ~2-5MB        less than 20ms
Decision engine        0KB (rules)   less than 1ms
Total pipeline         ~10MB         less than 50ms end-to-end

This is already lightweight. The current TF-IDF + XGBoost pipeline is
deployment-ready for on-device use without any modification.

---

## Deployment Architecture

Mobile App (iOS or Android)
|
├── Input Layer
|   ├── Journal text (typed or voice-to-text)
|   └── Sensor metadata (sleep from HealthKit or Google Fit,
|                        stress and energy from mat sensors)
|
├── Inference Layer (fully on-device)
|   ├── Text preprocessing (clean, TF-IDF transform)
|   ├── XGBoost classifier → predicted_state + confidence
|   ├── XGBoost regressor → predicted_intensity
|   └── Decision engine → what_to_do + when_to_do
|
└── Output Layer
    ├── Wellness recommendation card
    ├── Uncertainty indicator (low confidence → prompt for more input)
    └── Supportive message (template-based, no LLM needed)

---

## Model Format for Mobile

Framework        Format       Size      Notes
scikit-learn     joblib       ~500KB    TF-IDF vectorizer and scaler
XGBoost          .ubj/JSON    2-5MB     Native XGBoost mobile format
iOS              CoreML       ~3MB      Convert with coremltools
Android          ONNX         ~4MB      Convert with skl2onnx or xgboost ONNX

Conversion example:
    import joblib
    joblib.dump(tfidf, 'tfidf.joblib')
    clf.save_model('classifier.ubj')
    reg.save_model('regressor.ubj')

Total on-device storage: under 15MB — well within mobile app budgets.

---

## Upgrade Path — Lightweight Embeddings

If TF-IDF proves insufficient for short or complex texts, the next step is
replacing it with a small local sentence embedding model:

Model               Size    Latency     Notes
all-MiniLM-L6-v2    22MB    ~80ms       Best quality/size tradeoff
all-MiniLM-L3-v2    17MB    ~40ms       Faster, slightly lower quality
TinyBERT            56MB    ~150ms      Overkill for this use case

MiniLM-L6 is the recommended upgrade. It runs fully on-device, requires no
API, and handles negation and qualification far better than TF-IDF.

---

## Latency Budget

Target: under 200ms total (feels instant to the user)

Step                            Estimated time
Text input capture              0ms (already done)
Preprocessing and TF-IDF        5-10ms
XGBoost inference (both)        20-40ms
Decision engine                 less than 1ms
UI render                       16ms (one frame)
Total (TF-IDF pipeline)         ~50ms
Total (MiniLM upgrade)          ~150ms

Both are within the 200ms budget.

---

## Handling Edge Cases On-Device

Very short text ("ok", "fine"):
    uncertain_flag = 1
    App prompts: "Tell us a little more — how are you feeling right now?"
    Fallback recommendation: light_planning or pause

Missing metadata (sensor not connected):
    Use default values (sleep=6, stress=3, energy=3)
    Increase uncertain_flag sensitivity (threshold 0.35 instead of 0.40)
    Note lower confidence in UI

No internet connection:
    System runs entirely offline — this is the default design
    No degradation in functionality whatsoever

Low battery mode:
    Skip TF-IDF and XGBoost entirely
    Use decision engine only with raw metadata
    Latency drops to less than 2ms, still produces a useful recommendation

Distress keyword detection:
    Keyword list runs as a simple string match — zero latency, zero model cost
    Catches high-risk signals even if the ML model misclassifies

---

## Privacy

All inference happens on-device.
Journal text never leaves the device.
No user data is sent to any server.
Model weights are bundled with the app — no download required after install.

This is not just a technical choice. For a wellness product where users write
personal reflections, local inference is a trust requirement.

---

## Battery and Memory

TF-IDF + XGBoost inference  : ~0.1% battery per prediction
Memory footprint            : ~30MB RAM during inference
Background processing       : not required — inference runs only on user action

---

## Tradeoffs Summary

Approach          Size     Latency    Accuracy    Privacy
TF-IDF + XGBoost  ~10MB    ~50ms      Moderate    Full (on-device)
MiniLM + XGBoost  ~32MB    ~150ms     Better      Full (on-device)
Cloud API         ~0MB     ~800ms     Best        None (data leaves device)

For ArvyaX, the TF-IDF pipeline is the right starting point.
MiniLM is the natural upgrade when accuracy needs to improve.
Cloud APIs are never appropriate for a private wellness journal product.

---

## Summary

The current TF-IDF + XGBoost system is already edge-deployable.
Total model size is under 15MB. Inference latency is under 50ms.
The system runs fully offline. User data stays on device.

If richer text understanding is needed, MiniLM-L6 (22MB) is the recommended
upgrade — it stays within mobile constraints while handling the complex
language patterns that TF-IDF misses.

The decision engine is pure rule-based logic — zero size, zero latency,
fully explainable. It runs on any hardware including low-end devices.