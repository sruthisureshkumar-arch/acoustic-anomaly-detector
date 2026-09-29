# Live demo script (~2 minutes)

Rehearse this end to end at least once before presenting — the timing assumes the
baseline profile for your demo machine is **already recorded and saved** beforehand;
don't record it live unless you have time to spare.

**If you don't have the actual Snapdragon hardware for this recording** (e.g. filming on
a dev machine because the target device wasn't available in time): don't fake the NPU
label or benchmark number. Say plainly what you're showing instead — "this is running on
CPU fallback right now since I'm recording on a dev machine; the app auto-detects and
would use the Hexagon NPU on the actual Snapdragon target, and here's the code that does
that [point at `embedding.py`'s provider selection or the provider label showing
`CPUExecutionProvider`]." Judges evaluating Technical Implementation care that the
engineering is real and correct, which the rest of this demo still proves regardless of
which provider is active — an honest "here's the one thing I couldn't test without the
physical device" reads better than a claim that falls apart under a follow-up question.

A recorded video (60-90s) of this exact flow, run against a real machine with a real
induced fault (not a synthetic test tone), is worth recording separately and including in
the submission even if you also present live — every comparable-hackathon winner we
looked at (see README.md's "How this maps to the judging criteria") had one, and it's
insurance against something going wrong on stage. Screen-record the app plus a
phone/webcam angle on the actual machine, so judges can see the real-world trigger and
the app's reaction in the same shot.

**Picking a real fault to induce** (much more convincing than a phone playing a
recording): anything that changes a running machine's sound without being risky or
requiring specialized equipment — un-securing something so it rattles, adding light
friction/resistance to a moving part, briefly changing a fan/motor's load or speed,
loosening a panel so it buzzes. The point isn't to actually damage anything, just to
produce a genuinely different sound signature the app hasn't seen in its baseline.

## Before you go on stage

- [ ] `models/yamnet_npu.onnx` is the real AI-Hub-compiled asset (NOT `yamnet_public.onnx`)
- [ ] `python src/main.py` launches cleanly. On real Snapdragon hardware, the left panel
      should show `Inference: NPU (QNN) — model: yamnet_npu.onnx` — screenshot this, it's
      your proof. On a dev machine it'll correctly show `CPUExecutionProvider` instead --
      that's expected, not a bug, since QNN only exists on Snapdragon hardware.
- [ ] Click "Benchmark NPU vs CPU" once beforehand. On Snapdragon hardware this shows a
      real speedup number (e.g. "3.1 ms/window • 5.4x faster than CPU") — screenshot this,
      re-running it live on stage also works and is a good beat on its own (see below). On
      a dev machine it'll honestly report only one provider was available -- don't present
      that as an NPU number.
- [ ] A saved profile exists for the demo machine, with a sensible threshold (test it
      beforehand so you know roughly what triggers it)
- [ ] You have a way to make the "anomaly" sound happen on demand — ideally a real,
      physically induced change in the machine's sound (see above); a phone playing a
      recording of a faulty sound is an acceptable fallback if you can't safely alter the
      machine live
- [ ] Laptop is genuinely offline (airplane mode) for the "no internet needed" beat

## Script

**0:00 – 0:20 — The problem**
"Small workshops and independent mechanics can't afford industrial IoT sensor platforms
for predictive maintenance — those are built for factories with sensor budgets. But most
equipment problems show up in sound before they show up anywhere else. This app turns a
laptop's own microphone into that sensor, entirely offline."

**0:20 – 0:40 — Show it's really offline and really on the NPU**
Show airplane mode is on. Point at the `Inference: NPU (QNN)` label in the app, then the
"Performance proof" panel below it. "No cloud calls, no internet connection, and this
isn't just a label — we benchmarked it: [read the number, e.g. 'the NPU is running this
5x faster than CPU on this exact machine']." If time allows, click "Benchmark NPU vs CPU"
live here instead of just pointing at a cached number — it re-runs in a few seconds and
is a more convincing, harder-to-fake beat than a static screenshot.

**0:40 – 1:10 — Normal operation**
Select the pre-recorded profile, click Start monitoring. Let the machine run normally for
15-20 seconds. Point at the waveform, the live anomaly score gauge sitting comfortably
under the threshold line, and the green status light. "It's not just listening — it
learned this specific machine's normal sound in advance, so it's not relying on a generic
'this sounds like a machine problem' model trained on someone else's equipment."

**1:10 – 1:40 — Trigger the anomaly**
Introduce the anomalous sound (play the faulty recording near the mic, or physically
change the machine's sound). Narrate while the smoothed score climbs: "It's deliberately
not reacting to a single spike — a door slam or a cough shouldn't trigger a false alarm.
It waits for the deviation to *sustain* for a few consecutive windows." Let the alert
fire: status light goes red, alert sound plays, event logged with a timestamp and score.

**1:40 – 2:00 — Close**
Click "false alarm" or "confirmed issue" on the event to show the feedback loop. "Every
false alarm marked here widens the threshold slightly, so it adapts to this specific
machine and environment over time, without ever sending a single sample anywhere."

## If something goes wrong live

- **No alert triggers**: you rehearsed with too subtle a change, or the threshold is too
  wide from prior false-alarm feedback — have a second, more obviously different sound
  clip ready as a fallback.
- **False alarm during "normal" section**: don't panic-explain it away at length; briefly
  note "that's exactly the kind of false alarm the feedback button is for" and mark it,
  which doubles as an unplanned demonstration of that feature.
- **QNN doesn't show as active** (falls back to CPU on the actual demo hardware): still
  finish the demo — CPU fallback is a deliberate, working feature, not a bug — but be
  ready to explain why (driver/runtime mismatch, wrong target compile) rather than
  pretending it's not happening.
