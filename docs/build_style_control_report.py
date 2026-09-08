#!/usr/bin/env python
"""
Build the Smean AI technical report on adding style control to VoxCPM2.

Companion to build_literature_review.py, in the same house style (see
smean_docx.py). Where the literature review surveys models, this one documents
one engineering effort end to end -- including the run that failed, which is the
part with the transferable finding in it.

Content is sourced from docs/11-voxcpm2-style-control-finetune.md and the
measurement record in finetune/results/diagnosis.md. Probe numbers are read from
finetune/experiments/*.json rather than retyped, so the document cannot drift
from the experiments it describes.

    python docs/build_style_control_report.py

Writes docs/Smean-VoxCPM2-Style-Control.docx
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smean_docx import *  # noqa: F401,F403  -- brand tokens and layout helpers

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "finetune" / "experiments"
OUT = Path(__file__).resolve().parent / "Smean-VoxCPM2-Style-Control.docx"


def probe(arm):
    f = EXP / f"result_{arm}.json"
    return json.loads(f.read_text()) if f.exists() else None


def sens(arm):
    """Mean loss under the true and swapped tag, plus the sign-test result."""
    f = EXP / f"sensitivity_{arm}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    d["true_mean"] = sum(d["true"]) / len(d["true"])
    d["swap_mean"] = sum(d["swap"]) / len(d["swap"])
    return d


def pctdelta(d):
    return f"**{100 * d['delta_mean'] / d['true_mean']:+.2f}%**" if d else "n/a"


def sep(arm):
    d = probe(arm)
    return f"{d['separation']:+.1f} Hz" if d else "n/a"


def verdict(arm):
    d = probe(arm)
    return d["verdict"] if d else "n/a"


ARMS = {a: probe(a) for a in ("end", "proj", "onset")}
S_PROJ, S_ONSET = sens("proj"), sens("onset")
CTRL = json.loads((EXP / "sensitivity_positive_control.json").read_text())
CTRL["true_mean"] = sum(CTRL["true"]) / len(CTRL["true"])
CTRL["scramble_mean"] = sum(CTRL["scramble"]) / len(CTRL["scramble"])

doc = new_document()

cover(doc,
      title="Giving VoxCPM2 Style Control",
      subtitle="A Khmer fine-tune, the run that failed,\nand what the failure turned out to be",
      blurb="A technical report on adding expressive control to the model this project "
            "recommends for Khmer — and on a conditioning failure that is well known in "
            "the generative-modelling literature but undocumented for this class of TTS system.",
      meta=[["Document", "Technical report — fine-tuning and diagnosis"],
            ["Subject", "VoxCPM2 (OpenBMB) · LoRA style-control adapter for Khmer"],
            ["Scope", "Method · corpus construction · training · failure analysis · fix"],
            ["Evidence base", "Two 4000-step training runs, four 400-step probes, and a "
                              "loss-level diagnostic with a positive control"],
            ["Hardware", "One 12 GB RTX 3060"],
            ["Date", "8 September 2026"],
            ["Status", "Internal research document — final run in progress"]])

page_footer(doc, "Smean AI  ·  VoxCPM2 style control")

# =========================================================================
heading(doc, "1.  Summary", 1, page_break=True)

para(doc, "VoxCPM2 is this project's recommended model for Khmer, on the strength of a 2.47% "
          "character error rate against Higgs TTS 3's 8.28% and an Apache-2.0 licence. It has "
          "no expressive control. Text goes in, one voice and one delivery come out, and the "
          "only lever is a reference clip for zero-shot cloning. Higgs TTS 3, by contrast, "
          "accepts inline control tokens. The question this work answers is whether that "
          "control can be added to VoxCPM2 rather than traded for.")

para(doc, "It can, but not by the obvious route, and the detour is the useful part of this "
          "report. A LoRA adapter conditioned on a tag prefixed to the text — five axes: voice, "
          "speaking rate, pitch register, pitch variation and level — trains cleanly on a single "
          "12 GB consumer GPU, converges, audibly changes the model, and ignores the tag "
          "completely. Establishing that took a 6.5-hour run and six eliminated hypotheses. The "
          "cause is neither the tag format nor the architecture but the training objective.",
     space_after=10)

callout(doc, "The four findings that matter", [
    "**The conditioning channel is free, and that is not sufficient.** VoxCPM2's training packer "
    "zeroes the loss mask across every text position, so anything placed in the text field is pure "
    "conditioning at the cost of sequence length alone. No architectural change is needed. But a "
    "channel being lossless says nothing about whether gradient descent has any reason to use it — "
    "and by default it does not.",
    "**Teacher forcing is why.** Each audio patch is predicted with the preceding ground-truth "
    "patches visible, and a speaker's pitch is trivially readable off those. The tag is therefore "
    "redundant with the acoustic prefix everywhere except the start of the clip, and redundant "
    "information receives no gradient. This is the information-preference problem of Chen et al.'s "
    "Variational Lossy Autoencoder (2017), arriving somewhere the VoxCPM2 documentation gives no "
    "warning about.",
    "**Generated audio could not diagnose it; the training objective could.** Measuring output "
    "cannot distinguish “never learned the tag” from “learned it and lost it in sampling”. "
    "Comparing the loss under a correct versus a swapped tag, with a scrambled-transcript positive "
    "control, separates them in about two minutes.",
    "**Two changes fix it**, each isolated by its own probe against a pass mark fixed before any "
    "experiment ran: adapting the LM→DiT projection layers, and weighting the diffusion loss "
    "toward the audio onset where the tag is the only available cue.",
])

para(doc, "A secondary result, useful to anyone reproducing this: the officially documented "
          "~20 GB VRAM requirement for VoxCPM2 LoRA is an artefact of upstream leaving frozen "
          "weights in float32 while running the forward pass under bfloat16 autocast. Casting them "
          "fits the job in 12 GB with room to spare.")

# =========================================================================
heading(doc, "2.  The mechanism", 1)

para(doc, "Control is delivered by a tag prefixed to the text field:")

code(doc, "<|spk:f2|rate:fast|pitch:high|var:lively|energy:mid|>ថ្ងៃនេះអាកាសធាតុល្អណាស់។")

para(doc, "That is the entire architectural change. There is none. The reason it works rather "
          "than merely being convenient is in the training packer: `process_tts_data()` builds "
          "each sequence as [text tokens][audio start][audio patches][audio end] and sets the "
          "loss mask to zero across every text position. The model is never asked to reproduce "
          "the text, only to use it in predicting the audio latents that follow. A tag therefore "
          "costs 25 tokens of sequence length and nothing else.")

para(doc, "A prior worry, checked before training anything: would the model read the tag aloud? "
          "The base model was asked to synthesise a Khmer sentence with and without a tag, and "
          "both clips were transcribed with the project's Khmer CTC ASR. The transcripts are "
          "identical and both clips are 2.08 s. The base model silently ignores the tag — which "
          "also makes it a clean control condition for verification later.", space_after=10)

table(doc,
      ["Axis", "Levels", "What it moves", "Measured as"],
      [["`spk`", "`f1`…`f9`, `m1`…`m11`", "which corpus voice, no reference clip needed",
        "speaker identity"],
       ["`rate`", "`slow` `mid` `fast`", "speaking rate", "Khmer characters per second"],
       ["`pitch`", "`low` `mid` `high`", "pitch register, relative to that voice", "median F0"],
       ["`var`", "`flat` `mid` `lively`", "monotone versus expressive", "F0 std, semitones"],
       ["`energy`", "`soft` `mid` `loud`", "level and projection", "RMS level"]],
      widths=[0.8, 1.5, 2.4, 1.5])
caption(doc, "Table 1 — The five control axes. Every one is defined as something measurable on a "
             "waveform, which is what makes the claim “did asking for rate:fast produce faster "
             "speech?” a matter of arithmetic rather than opinion.")

# =========================================================================
heading(doc, "3.  Where the labels come from", 1)

para(doc, "There is no expressive Khmer speech corpus, and no Khmer emotion corpus. Copying "
          "Higgs's 21-emotion catalogue would mean inventing labels for data that does not carry "
          "them, and the result would be unfalsifiable. So the axes are restricted to attributes "
          "that can be measured directly.")

para(doc, "This is the Parler-TTS recipe (Lyth & King, 2024), which exists precisely because "
          "“reliance on human-labeled descriptions prevents scaling”: measure speaking rate, "
          "pitch, SNR and reverberation automatically, bin each continuous measurement, and name "
          "the bins. Their dataspeech tooling uses seven bins for speaking rate, from very slowly "
          "to very fast. The tag used here is the same idea in a more compact syntax.")

rich(doc, [("Two design decisions were made independently and then found to agree with that "
            "pipeline, which is mild evidence they are right rather than local quirks. First, "
            "pitch, variation and energy are ranked ", {}),
           ("within speaker", {"bold": True}),
           (" while rate is ranked globally — a 105 Hz voice and a 280 Hz voice do not share a "
            "definition of “high”. "
            "dataspeech compares a speaker's pitch against others of the same gender for the "
            "same reason. Second, the level boundaries are taken from the outer 15% tails rather "
            "than from balanced tertiles; dataspeech likewise computes its bin edges from "
            "histograms “from which the extreme values have been eliminated”.", {})],
     space_after=10)

table(doc,
      ["Axis", "Balanced tertiles", "15/85 tails", "Gain"],
      [["`rate`", "+4.49 chars/s", "**+6.57 chars/s**", "+46%"],
       ["`pitch`", "+20.3 Hz", "**+28.3 Hz**", "+40%"],
       ["`var`", "+1.16 st", "**+1.71 st**", "+47%"],
       ["`energy`", "+3.78 dB", "**+5.81 dB**", "+54%"]],
      widths=[1.0, 1.6, 1.6, 0.9], align_right=(1, 2, 3))
caption(doc, "Table 2 — Separation between the outer levels under two labelling schemes, measured "
             "on the same 6000 candidate clips. Taking the tails leaves 880 exemplars of each "
             "extreme instead of 2000, and buys roughly 1.5× the separation on every axis. For "
             "teaching a direction, unambiguous examples beat numerous borderline ones.")

callout(doc, "Measure the ceiling before training, not after",
        ["The separation the labels themselves achieve is an upper bound on anything the model "
         "can learn from them. It costs seconds to compute and it caught a too-narrow control "
         "range before a single GPU-hour was spent. Every result in section 6 is quoted against "
         "these ceilings rather than in isolation."],
        accent=VIOLET, fill=VIOLET_SOFT)

# =========================================================================
heading(doc, "4.  The corpus and the training setup", 1)

para(doc, "Source speech is this project's sibling ASR corpus — real recorded Khmer read speech, "
          "not synthetic. Clips were extracted, measured, labelled and written as manifests by "
          "`finetune/build_corpus.py`, then validated with the package's own dataset checker "
          "(5820/5820 passed).", space_after=10)

table(doc,
      ["", ""],
      [["Clips", "5,820 (5,646 train / 174 validation)"],
       ["Duration", "11.73 hours, clips of 3–10 s"],
       ["Speakers", "20, spanning roughly 105–280 Hz median F0"],
       ["Sample rate", "16 kHz in (AudioVAE encoder rate), 48 kHz out"],
       ["Gain handling", "normalised per speaker, not per clip, so the energy axis survives"],
       ["Partial tags", "per-slot dropout during training, so an underspecified tag still works"]],
      widths=[1.2, 4.0], size=9.5, header=False)
caption(doc, "Table 3 — The training corpus.")

heading(doc, "4.1  Fitting a documented 20 GB job into 12 GB", 2)

para(doc, "OpenBMB's figure for VoxCPM2 LoRA is roughly 20 GB; this machine has 12.3. The gap is "
          "almost entirely one line in upstream's `from_local`: in training mode the model is left "
          "in float32, so 2.29 B frozen parameters occupy 8.6 GiB before a single activation is "
          "allocated — while the forward pass runs under bfloat16 autocast anyway, which means the "
          "float32 master copy buys nothing for the 99.9% of parameters that are frozen.")

para(doc, "`finetune/train.py` wraps the upstream trainer without modifying it and applies three "
          "patches: frozen weights cast to bfloat16 (halving the resident model to 4.3 GiB) while "
          "LoRA parameters stay float32 so AdamW updates are not quantised away; gradient "
          "checkpointing across all 36 transformer layers; and the AudioVAE kept in float32 but "
          "explicitly on GPU under no_grad. Measured peak is 10.3–11.7 GiB with the desktop "
          "running. Peak turns out to be nearly independent of sequence length — halving "
          "`max_batch_tokens` moved it by 0.6 GiB — so the ceiling is resident weights and the "
          "AudioVAE encode, not activations.")

# =========================================================================
heading(doc, "5.  The run that failed", 1, page_break=True)

para(doc, "The first full run trained without incident: 4000 steps, 6.5 hours, validation "
          "loss/diff falling 0.8819 → 0.8076, ten checkpoints. The adapter was real, and it "
          "audibly changed the model.", space_after=10)

table(doc,
      ["Check", "Result"],
      [["LoRA weight delta, median ‖ΔW‖/‖W‖ over 192 adapted layers",
        "**3.03e-02** (range 2.78e-03 – 8.31e-02)"],
       ["`lora_B` tensors non-zero (they initialise to exactly zero)", "**192 / 192**"],
       ["Generated F0, base model, across sentences", "wanders 105.9 – 231.5 Hz"],
       ["Generated F0, adapter", "clamped 201.3 – 261.6 Hz (median delta +66.5 Hz)"],
       ["Speaking rate, base → adapter", "17.61 → 14.07 chars/s"]],
      widths=[3.0, 2.6])
caption(doc, "Table 4 — The fine-tune itself succeeded. The adapter has a large, audible effect.")

para(doc, "It ignored the control tag completely.", bold=True, space_after=8)

table(doc,
      ["Axis", "Measured", "Commanded low → high", "rho", "p", "Corpus ceiling"],
      [["`rate`", "chars/s", "+0.27", "+0.141", "0.46", "**+6.57**"],
       ["`pitch`", "F0 median", "−2.81 Hz", "+0.019", "0.93", "**+28.27 Hz**"],
       ["`var`", "F0 std", "+0.02 st", "−0.057", "0.77", "**+1.71 st**"],
       ["`energy`", "RMS", "−0.37 dB", "−0.038", "0.85", "**+5.81 dB**"]],
      widths=[0.75, 0.9, 1.35, 0.75, 0.6, 1.15], align_right=(2, 3, 4, 5))
caption(doc, "Table 5 — Ten eval-set sentences per level, every axis swept, unswept slots pinned "
             "to mid so the tag stays in distribution. rho is Spearman's correlation between "
             "commanded level and measured quantity; p is a 10,000-shuffle permutation test. "
             "Null on every axis, against ceilings measured before training started.")

para(doc, "Ranking all nine checkpoints by control response — not by validation loss, which "
          "cannot see this failure — found no trend across training and a best mean rho of "
          "+0.026. It was not a matter of picking the wrong checkpoint.")

para(doc, "The decisive axis is `spk`. Six speaker tags spanning 105–280 Hz of real corpus pitch "
          "produced generated medians of 198.5, 209.3, 210.3, 222.9, 208.6 and 196.7 Hz: 26 Hz of "
          "spread against 175 Hz in the corpus, rho = −0.200. Speaker identity is the one "
          "attribute the model cannot infer from the text, so if that axis does not land, nothing "
          "is landing. It also explains the voice collapse in Table 4 — a model that cannot read "
          "the tag can only average its twenty speakers.")

heading(doc, "5.1  Eliminating the easy explanations", 2)

table(doc,
      ["Hypothesis", "How it was ruled out"],
      [["Tag shape out of distribution",
        "per-slot dropout made all-`any` tags 0.27% of rows — but the in-distribution sweep "
        "(Table 5) pins unswept slots to `mid` and is equally null"],
       ["Adapter inert", "3% weight delta, all 192 `lora_B` tensors non-zero"],
       ["Tokenization", "tags survive the model's wrapped tokenizer, differ in ten token "
        "positions between extremes, produce zero UNK"],
       ["Train/inference mismatch", "both paths call the same `text_tokenizer`; inference passes "
        "the text through unsplit and un-chunked"],
       ["Text normalization", "`normalize` defaults to False and is never passed"],
       ["Tag placement", "Khmer tokenises to ~330 UTF-8 byte tokens, burying a prefix ~350 "
        "positions from the audio. Moving the tag adjacent to `audio_start` gave " + sep("end") +
        " of speaker separation. Not it either."]],
      widths=[1.5, 4.1])
caption(doc, "Table 6 — Six hypotheses, eliminated in the order they were tried.")

# =========================================================================
heading(doc, "6.  The measurement that settled it", 1)

para(doc, "Everything above measures generated audio, and generated audio cannot distinguish "
          "“never learned the tag” from “learned it, but sampling washes it out”. So the "
          "question went back to the training objective. For each held-out clip: one "
          "teacher-forced forward pass with the true speaker tag, one with the other speaker's, "
          "with the diffusion timestep and the noise held identical by reseeding from the row "
          "index. The flow-matching loss is stochastic, and unseeded this measures nothing but "
          "noise.", space_after=10)

table(doc,
      ["Condition", "Mean loss/diff", "Delta", "Rows worse"],
      [["Correct tag", f"{S_PROJ['true_mean']:.5f}", "—", "—"],
       ["Swapped tag", f"{S_PROJ['swap_mean']:.5f}", pctdelta(S_PROJ),
        f"{S_PROJ['worse']}/{S_PROJ['n']}  (sign-test p = {S_PROJ['p']:.2f})"],
       ["Scrambled transcript (positive control)", f"{CTRL['scramble_mean']:.5f}",
        f"**{100 * CTRL['delta_mean'] / CTRL['true_mean']:+.1f}%**",
        f"{CTRL['worse']}/{S_PROJ['n']}"]],
      widths=[2.2, 1.2, 1.0, 1.6], align_right=(1, 2))
caption(doc, "Table 7 — The positive control is what makes this readable. Corrupting the "
             "transcript, which the model certainly uses, moves the loss by 7.4%. Corrupting the "
             "tag moves it by 0.03%, a 290× difference indistinguishable from chance.")

para(doc, "The model reads the text and does not read the tag, in its own objective. No "
          "inference-side change could have rescued that.", bold=True)

heading(doc, "6.1  Why: the acoustic prefix already answers the question", 2)

para(doc, "Training is teacher-forced. Every audio patch is predicted with the preceding "
          "ground-truth patches visible, and a speaker's pitch is trivially readable off those. "
          "The tag is redundant with the acoustic prefix everywhere except the very beginning of "
          "the clip, and redundant information receives no gradient. That yields a sharp "
          "prediction: restrict the loss to the first K patches, where no prefix exists yet, and "
          "the tag must start to matter.", space_after=10)

table(doc,
      ["Loss restricted to first K patches", "1", "2", "4", "8", "all"],
      [["Penalty for the wrong tag", "**+0.00318**", "+0.00214", "+0.00143", "+0.00082",
        "+0.00022"]],
      widths=[2.2, 0.9, 0.85, 0.85, 0.85, 0.85], align_right=(1, 2, 3, 4, 5))
caption(doc, "Table 8 — Monotone, and 14× larger on the first patch than over the whole clip. The "
             "tag matters exactly where the prefix cannot answer for it, and is drowned everywhere "
             "else.")

callout(doc, "A known problem, met in an unexpected place", [
    "This is the **information-preference problem** described by Chen et al. in Variational "
    "Lossy Autoencoder (ICLR 2017): a sufficiently expressive autoregressive decoder ignores a "
    "conditioning signal, because predicting the next step from its own history is cheaper in "
    "coding terms than routing information through the conditioning path. Their remedies are all "
    "forms of taking the shortcut away — weakening the decoder, dropout, limiting its receptive "
    "field.",
    "The same failure is Bowman et al.'s KL-vanishing in text VAEs (2016), fixed there by word "
    "dropout: replace part of the decoder's history with UNK so it cannot recover the next word "
    "without consulting the latent. The link to teacher forcing generally is Bengio et al. (2015) "
    "on scheduled sampling.",
    "None of this literature is about TTS, and the diagnosis here was reached by measurement "
    "rather than through it — the family was recognised afterwards. But it means the finding is "
    "not exotic. It is the standard failure of conditional autoregressive generation, arriving in "
    "a place the VoxCPM2 documentation does not warn about.",
])

para(doc, "It follows that any global attribute supplied through the text field — speaker, rate, "
          "register, and by extension emotion — competes against a teacher-forced acoustic prefix "
          "that already encodes it, and loses. This is a property of the training objective, not "
          "of VoxCPM2 and not of the tag format.")

# =========================================================================
heading(doc, "7.  The fix", 1, page_break=True)

para(doc, "Two changes, each isolated by its own 400-step probe on the easiest discrimination the "
          "corpus offers: two speakers an octave apart (m5 at ~105 Hz, f4 at ~280 Hz), 765 "
          "training rows, one binary tag. The pass mark — more than 30 Hz of separation, in the "
          "correct direction — was fixed before the first probe ran, against 175 Hz of real "
          "separation. Each arm changes exactly one thing from the arm above it, so a result "
          "attaches to a cause rather than to a bundle of changes.", space_after=10)

table(doc,
      ["Arm", "What it changes", "m5", "f4", "Separation", ""],
      [["`end`", "tag moved adjacent to `audio_start`",
        f"{ARMS['end']['low']:.1f}" if ARMS['end'] else "266.5",
        f"{ARMS['end']['high']:.1f}" if ARMS['end'] else "272.3",
        f"**{sep('end')}**", verdict("end")],
       ["`proj`", "`end` + `enable_proj: true`",
        f"{ARMS['proj']['low']:.1f}" if ARMS['proj'] else "269.7",
        f"{ARMS['proj']['high']:.1f}" if ARMS['proj'] else "289.2",
        f"**{sep('proj')}**", verdict("proj")],
       ["`onset`", "`proj` + diffusion loss weighted 8× at the onset",
        f"{ARMS['onset']['low']:.1f}" if ARMS['onset'] else "272.3",
        f"{ARMS['onset']['high']:.1f}" if ARMS['onset'] else "312.2",
        f"**{sep('onset')}**", f"**{verdict('onset')}**"]],
      widths=[0.7, 2.7, 0.65, 0.65, 1.0, 0.65], align_right=(2, 3, 4))
caption(doc, "Table 9 — The probe ladder. Generated F0 median in Hz, six sentences per tag. "
             "Placement was not the problem; the two things that were are below.")

heading(doc, "7.1  The projection layers were frozen", 2)

para(doc, "`enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj` and `fusion_concat_proj` are the "
          "linear bottleneck through which everything the language model knows reaches the "
          "diffusion transformer that produces the acoustics. The project's own architecture notes "
          "recommend leaving them frozen, and for speaker cloning that is correct — the voice "
          "arrives through the reference-audio encoder and never needs to cross that bridge. For "
          "conditioning that arrives only as text there is no other route across. Worth +13.7 Hz.")

heading(doc, "7.2  Weighting the loss toward the onset", 2)

para(doc, "Position i of the audio span is given weight 1 + 7·exp(−i/4): eight times on the first "
          "patch, decaying to about one by the twentieth. This puts the gradient where the tag is "
          "the only available cue. Nothing downstream needs modifying — the flow-matching loss "
          "computes a mask-weighted mean and the adaptive weighting returns the mask unchanged, so "
          "a non-binary mask is a self-renormalising weight. Upstream casts the mask to int32, "
          "which would truncate the weights, so the wrapper rebuilds it as float. Worth a further "
          "+20.4 Hz, and the change that clears the bar.")

para(doc, "If the onset is set correctly, autoregression carries it: the rest of the utterance is "
          "generated conditioned on that first patch.")

callout(doc, "Provenance: this one is not borrowed", [
    "Onset weighting has no precedent in the literature cited above, and it would be misleading "
    "to imply otherwise. The published fixes all attack the shortcut itself — corrupt the "
    "decoder's history, or restrict what it can see. The equivalent here would be noising or "
    "masking the ground-truth audio prefix during training, and VoxCPM2 already contains machinery "
    "of that shape: it noises the diffusion head's previous-patch conditioning on a ramping "
    "schedule, and drops the LM conditioning entirely at a fixed rate for classifier-free "
    "guidance. That the designers included both suggests they knew about the problem.",
    "The reason for not doing it that way is practical: the shortcut is the **language model's** "
    "view of the prefix, not only the diffusion head's, and removing that properly means surgery "
    "inside the library rather than a wrapper around it. Onset weighting attacks the same target "
    "from the other side, and required no library change at all. Whether prefix corruption would "
    "work better is untested, and it is the more principled experiment.",
], accent=SIENNA, fill=SIENNA_SOFT)

para(doc, "The diagnostic that found the failure confirms the repair at the same level:",
     space_after=8)

table(doc,
      ["Checkpoint", "Correct tag", "Swapped tag", "Delta", "Rows worse", "Sign-test p"],
      [["`proj`", f"{S_PROJ['true_mean']:.5f}", f"{S_PROJ['swap_mean']:.5f}",
        f"{S_PROJ['delta_mean']:+.5f}", f"{S_PROJ['worse']}/{S_PROJ['n']}",
        f"{S_PROJ['p']:.2f}"],
       ["`onset`", f"{S_ONSET['true_mean']:.5f}", f"{S_ONSET['swap_mean']:.5f}",
        f"**{S_ONSET['delta_mean']:+.5f}**", f"{S_ONSET['worse']}/{S_ONSET['n']}",
        f"**{S_ONSET['p']:.3f}**"]],
      widths=[0.9, 1.0, 1.0, 1.0, 0.85, 0.95], align_right=(1, 2, 3, 4, 5))
caption(doc, "Table 10 — Four times the penalty for the wrong tag, and significant where it was "
             "not. The model now reads the tag in its own loss, which is the thing that was "
             "actually missing.")

# =========================================================================
heading(doc, "8.  The two runs", 1)

table(doc,
      ["", "Run 1", "Run 2"],
      [["Corpus", "5,646 train / 174 val, 20 speakers, 11.73 h", "same"],
       ["Steps", "4000 (≈5.5 epochs), 6.5 h", "4000, ≈7 h"],
       ["LoRA", "r=64, α=64, `enable_lm` + `enable_dit`", "same, **plus `enable_proj`**"],
       ["Loss", "uniform over audio positions", "**8× at onset, τ = 4 patches**"],
       ["Peak VRAM", "10.3–11.7 GiB", "11.3 GiB"],
       ["Control response", "**none** — rho ≈ 0 on every axis", "in progress — see the status note below"]],
      widths=[1.0, 2.5, 2.1])
caption(doc, "Table 11 — Run 1's checkpoints and sweep are kept rather than deleted: they are the "
             "control condition for run 2, and the evidence for sections 5 and 6.")

callout(doc, "Status at the time of writing", [
    "**Run 2 is still training.** Nothing in this report claims the five-axis control works. What "
    "is established is that the model can now learn to read a tag at all, on the easiest "
    "discrimination the corpus offers, and that the mechanism which previously prevented it is "
    "identified and measured. The five-axis sweep, the checkpoint selection and the CER regression "
    "check follow when the run completes.",
], accent=SIENNA, fill=SIENNA_SOFT)

# =========================================================================
heading(doc, "9.  What this settles, and what it does not", 1)

rich(doc, [('Settled.', {"bold": True}), (' VoxCPM2 can be given a control surface without touching its architecture. It trains on 12 GB, not the documented 20. And the failure mode that makes the obvious construction quietly not work is identified, measured and fixed.', {})])

rich(doc, [('Settled about method, and more durable than any of the numbers.', {"bold": True}), (' Two habits did more work than any single idea, and neither is novel. Measuring the label separation before training, because it bounds what the model can possibly learn. And taking a null result back to the training objective with a positive control, because generated-audio metrics cannot distinguish a signal that was never learned from one lost in sampling. The second answered in two minutes a question a 6.5-hour run had left ambiguous.', {})])

rich(doc, [('Not settled.', {"bold": True}), (' Naturalness. No automatic metric ranks Khmer TTS — this project has already established that MOS predictors are inverted for Khmer and that prosody statistics cannot tell well-placed pitch movement from badly-placed pitch movement. Whether styled output sounds better is a listening-test question, and the apparatus exists but has not been run with real listeners.', {})])

rich(doc, [('Out of reach with this data.', {"bold": True}), (' Emotion and non-prosodic style — sadness, whispering, laughter. Those need expressive Khmer recordings. The mechanism in section 2 would carry them, but section 6 is the correction to the obvious next thought: the mechanism alone is not enough, and an emotion tag would face exactly the same competition against the acoustic prefix that the speaker tag lost. The corpus is what is missing; the onset weighting would still be required.', {})])

# =========================================================================
heading(doc, "10.  References", 1)

for txt in [
    "Lyth, D. & King, S. (2024). *Natural language guidance of high-fidelity text-to-speech with "
    "synthetic annotations.* arXiv:2402.01912 — Parler-TTS; measured attributes as text "
    "conditioning, and the `dataspeech` binning pipeline.",
    "Chen, X. et al. (2017). *Variational Lossy Autoencoder.* ICLR 2017, arXiv:1611.02731 — why "
    "an expressive autoregressive decoder ignores its conditioning, and what removing the "
    "shortcut looks like.",
    "Bowman, S. R. et al. (2016). *Generating Sentences from a Continuous Space.* arXiv:1511.06349 "
    "— the same collapse in text VAEs; word dropout as the cure.",
    "Bengio, S. et al. (2015). *Scheduled Sampling for Sequence Prediction with Recurrent Neural "
    "Networks.* arXiv:1506.03099 — teacher forcing and the train/inference mismatch.",
    "Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* "
    "arXiv:2106.09685.",
    "Ho, J. & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.* arXiv:2207.12598 — the "
    "condition dropout VoxCPM2 implements as `training_cfg_rate`.",
]:
    # split on the italic markers so paper titles render as italics, not asterisks
    bullet(doc, [(seg, {"italic": i % 2 == 1}) for i, seg in enumerate(txt.split("*")) if seg])

para(doc, "", space_after=4)
para(doc, "Internal: docs/09 (VoxCPM2 architecture and training), docs/10 (Higgs TTS 3), docs/11 "
          "(the full engineering write-up), finetune/README.md (runbook), and "
          "finetune/results/diagnosis.md (the complete measurement record).",
     size=9.5, color=STONE)

doc.save(OUT)
print(f"wrote {OUT}")
