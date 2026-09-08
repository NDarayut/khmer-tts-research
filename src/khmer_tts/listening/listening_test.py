"""
Build a blind, randomized A/B listening test -- the only valid way to measure
naturalness for Khmer.

WHY THIS AND NOT A METRIC

score_naturalness.py established that UTMOS is *inverted* for this comparison:
across 400 clips its rank correlation with Khmer CER is +0.55, and the clips it
scores highest are the ones that get the Khmer most wrong. Level and silence
differences were controlled for and are not the cause. DNSMOS behaves the same
way. prosody_stats.py can describe what differs between two voices, but
description is not judgement -- and its own built-in check shows pitch variance
does not recover the known verdict on `mms`.

So naturalness has to be measured on human ears. That is not a fallback; for
naturalness it is the definition. What this file provides is the apparatus that
makes a listening result *evidence* rather than an impression.

WHAT MAKES THIS A TEST RATHER THAN A LISTENING SESSION

  Blind          Model identity never reaches the page's visible DOM; the key
                 is written into the results payload only.
  Randomized     Which model is on the left is a coin flip per trial, and trial
                 order is shuffled per session. Both matter: listeners favour
                 the left/first option otherwise, which would silently become
                 whichever model the loop happened to put first.
  Forced choice  2AFC, not a 1-5 MOS scale. Two systems compared head to head
                 need no scale calibration across listeners, and a preference
                 test detects a given difference in far fewer trials than
                 rating each system separately and differencing the means.
  Anchored       A fixed share of trials pit a contender against `fish-s2` or
                 `mms`, which are already known-bad by ear AND by CER. A
                 listener who does not reliably pick the contender in those
                 trials was not listening; analyse.py reports that per listener
                 and can exclude them. Without catch trials an unsupervised
                 remote test cannot distinguish careful listeners from clickers.
  One question   "Which sounds more natural" only. CER already answers
                 intelligibility, and asking two questions per trial roughly
                 doubles fatigue for the axis we are not short of data on. The
                 instructions tell listeners explicitly to ignore whether the
                 words are correct, because otherwise this silently re-measures
                 intelligibility -- where VoxCPM2 already wins, which would
                 manufacture agreement rather than test for it.

HOW MANY LISTENERS, HOW MANY SENTENCES

For a two-alternative forced choice against a 50% null, at alpha=0.05 and 80%
power, the trial counts are roughly:

    true preference   trials needed
        70%                47
        65%                85
        60%               194
        55%               776

Trials from one listener are correlated, so total trials overstate the real
power -- the honest unit of replication is the listener. The default is 40
sentences, because 5 listeners x 40 = 200 trials is the point where a 60%
preference becomes resolvable; at 30 sentences the same five listeners can only
resolve 65% and above. If the two systems are genuinely close, an underpowered
test returns "no significant difference", which is not the same finding as "no
difference" and is easy to misreport as one.

Five listeners is a reasonable floor; three is a pilot. One listener, however
expert, cannot separate a property of the model from a personal preference --
that is the specific gap this apparatus closes.

THE CONFOUND THIS TEST CANNOT REMOVE ON ITS OWN

Each model is running its own default voice, and they are not the same voice:
prosody_stats.py measures a median F0 of ~206 Hz for VoxCPM2 against ~114 Hz for
Higgs TTS 3 -- close to an octave apart, i.e. a different apparent speaker and
plausibly a different apparent gender. A listener asked which sounds more
natural may partly be answering which voice they prefer, and that is a property
of an arbitrary default, not of the synthesis.

Two ways to deal with it, in order of rigour:

  1. Re-synthesize both models with the SAME reference voice. Both support
     zero-shot cloning, so this is available and it removes the confound at the
     source. It costs a re-run of synthesize.py against one reference clip.
  2. Failing that, run the test as-is and treat a narrow result as unresolved.
     A large, consistent preference is still informative -- voice preference is
     unlikely to be worth 30 points across five listeners -- but a 55-60% result
     cannot be separated from timbre preference by this design.

Option 1 is the right experiment if the decision is close. This file does not
do it for you, because it scores the clips that already exist.

    python evaluation/listening_test.py
    python evaluation/listening_test.py --pair voxcpm2:higgs3 --sentences 30
    python evaluation/listening_test.py --anchors 6 --bitrate 64k

Writes evaluation/listening_test.html -- one self-contained file, no sibling
assets, openable from disk or emailable. Votes accumulate in the browser's
localStorage as they are cast, so a closed tab does not lose the session; at
the end the listener downloads a .json and sends it back. Score the collected
files with:

    python evaluation/listening_analyse.py results/*.json
"""

import argparse
import base64
import html
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from khmer_tts.common import ROOT, load_entries, wav_path  # noqa: E402

OUT_PATH = ROOT / "evaluation" / "listening_test.html"

# base64 inflates by 4/3; keep the page comfortably openable and emailable.
MAX_PAGE_BYTES = 12 * 1024 * 1024

# Known-bad by ear and by CER (78.01% and 25.12% median) -- see CLAUDE.md.
ANCHOR_MODELS = ("fish-s2", "mms")


def encode_mp3(path, bitrate):
    """-> base64 mono MP3, or None if the wav is missing / ffmpeg fails."""
    path = Path(path)
    if not path.is_file():
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-b:a", bitrate, str(tmp)],
            capture_output=True,
        )
        if result.returncode != 0 or not tmp.stat().st_size:
            return None
        return base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        tmp.unlink(missing_ok=True)


def pick_sentences(entries, n, seed):
    """Spread the sample across categories rather than taking the first N --
    prosody differences show up in questions and exclamations that a prefix
    of the set would under-sample."""
    rng = random.Random(seed)
    by_cat = {}
    for entry in entries:
        by_cat.setdefault(entry.get("category", "?"), []).append(entry)
    for bucket in by_cat.values():
        rng.shuffle(bucket)

    picked, cats = [], sorted(by_cat)
    while len(picked) < n and any(by_cat[c] for c in cats):
        for cat in cats:
            if by_cat[cat] and len(picked) < n:
                picked.append(by_cat[cat].pop())
    return picked


def build_trials(entries, pair, anchors, seed):
    """-> list of trial dicts. `side` records which model went left, so the
    analysis can check for a side bias rather than assume there is none."""
    rng = random.Random(seed + 1)
    model_a, model_b = pair
    trials = []

    for entry in entries:
        left_is_a = rng.random() < 0.5
        trials.append({
            "id": entry["id"],
            "kind": "test",
            "left": model_a if left_is_a else model_b,
            "right": model_b if left_is_a else model_a,
        })

    # Anchors reuse sentences already in the set, pairing a random contender
    # against a known-bad model. Same sentence, so the only difference is voice.
    anchor_pool = [e for e in entries]
    rng.shuffle(anchor_pool)
    for entry in anchor_pool[:anchors]:
        good = rng.choice(pair)
        bad = rng.choice(ANCHOR_MODELS)
        left_is_good = rng.random() < 0.5
        trials.append({
            "id": entry["id"],
            "kind": "anchor",
            "left": good if left_is_good else bad,
            "right": bad if left_is_good else good,
            "expected": good,
        })

    rng.shuffle(trials)
    return trials


PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Khmer TTS listening test</title>
<style>
  :root {{
    --teal:#0B776F; --teal-soft:#CCFBF1; --ink:#1A1915; --stone:#6B6A62;
    --oat:#E5E1D8; --linen:#F4F2EC; --bone:#FBFAF7; --paper:#fff;
    --violet:#6D28D9; --sienna:#C2410C;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bone); color:var(--ink);
         font:15px/1.55 "Inter Tight",system-ui,-apple-system,Segoe UI,sans-serif; }}
  .wrap {{ max-width:760px; margin:0 auto; padding:28px 20px 80px; }}
  h1 {{ font-size:23px; margin:0 0 4px; letter-spacing:-.01em; }}
  .sub {{ color:var(--stone); font-size:13.5px; margin:0 0 22px; }}
  .card {{ background:var(--paper); border:1px solid var(--oat); border-radius:12px;
           padding:20px; margin-bottom:16px; }}
  .brand {{ color:var(--teal); font-weight:700; letter-spacing:.06em; font-size:11px; }}
  .khmer {{ font-size:20px; line-height:1.85; margin:14px 0 18px;
            font-family:"Khmer OS System","Noto Sans Khmer",sans-serif; }}
  .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  @media (max-width:560px) {{ .pair {{ grid-template-columns:1fr; }} }}
  .opt {{ border:1.5px solid var(--oat); border-radius:10px; padding:14px; text-align:center;
          background:var(--linen); }}
  .opt h3 {{ margin:0 0 10px; font-size:13px; color:var(--stone); letter-spacing:.08em; }}
  audio {{ width:100%; }}
  button {{ font:inherit; cursor:pointer; border-radius:999px; border:1.5px solid var(--teal);
            background:var(--paper); color:var(--teal); padding:9px 18px; font-weight:600;
            margin-top:11px; width:100%; }}
  button:hover {{ background:var(--teal-soft); }}
  button.primary {{ background:var(--teal); color:#fff; }}
  button.ghost {{ border-color:var(--oat); color:var(--stone); width:auto; padding:7px 16px; }}
  .bar {{ height:5px; background:var(--oat); border-radius:99px; overflow:hidden; margin:6px 0 20px; }}
  .bar > i {{ display:block; height:100%; background:var(--teal); width:0; transition:width .25s; }}
  .meta {{ display:flex; justify-content:space-between; align-items:center;
           color:var(--stone); font-size:12.5px; }}
  ol {{ padding-left:20px; }} ol li {{ margin:7px 0; }}
  .warn {{ background:#FFEDD5; border-left:4px solid var(--sienna); padding:12px 14px;
           border-radius:0 8px 8px 0; font-size:13.5px; margin:14px 0; }}
  .note {{ background:var(--teal-soft); border-left:4px solid var(--teal); padding:12px 14px;
           border-radius:0 8px 8px 0; font-size:13.5px; margin:14px 0; }}
  code {{ background:var(--linen); padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  textarea {{ width:100%; height:150px; font-family:ui-monospace,Menlo,Consolas,monospace;
              font-size:11px; border:1px solid var(--oat); border-radius:8px; padding:10px; }}
  input[type=text] {{ font:inherit; padding:9px 12px; border:1.5px solid var(--oat);
                      border-radius:8px; width:100%; }}
  .hidden {{ display:none !important; }}
</style>
<div class="wrap">
  <div class="brand">SMEAN AI</div>
  <h1>Khmer TTS — blind listening test</h1>
  <p class="sub">Which voice sounds more natural? About {minutes} minutes.</p>

  <div id="intro" class="card">
    <h2 style="margin-top:0;font-size:17px">Before you start</h2>
    <ol>
      <li>Use <strong>headphones</strong> in a quiet room, and keep the volume steady throughout.</li>
      <li>You will hear the <strong>same Khmer sentence</strong> read by two different systems,
          A and B. Listen to both — you can replay as often as you like.</li>
      <li>Choose the one that sounds <strong>more natural</strong>: more like a real person
          speaking Khmer. Judge rhythm, intonation, stress and voice quality.</li>
      <li><strong>Ignore whether the words are correct.</strong> If one version misreads or skips
          a word, that is not what this test is measuring — we measure it separately. Judge only
          how natural the delivery sounds.</li>
      <li>There is no right answer. Trust your first impression; do not deliberate.</li>
    </ol>
    <div class="note">Which system is A and which is B changes randomly on every trial, and the
      order of trials is different for every listener. Nothing on this page tells you which
      system you are hearing — that is deliberate.</div>
    <label style="display:block;margin:16px 0 6px;font-weight:600;font-size:13.5px">
      Your name or initials <span style="color:var(--stone);font-weight:400">(so responses can be
      grouped by listener; it is not published)</span></label>
    <input type="text" id="listener" placeholder="e.g. Dara N." autocomplete="off">
    <button class="primary" id="start" style="margin-top:16px">Start the test</button>
  </div>

  <div id="test" class="hidden">
    <div class="meta"><span id="counter"></span><span id="cat"></span></div>
    <div class="bar"><i id="prog"></i></div>
    <div class="card">
      <div style="color:var(--stone);font-size:12px;letter-spacing:.08em">SENTENCE</div>
      <div class="khmer" id="sentence"></div>
      <div class="pair">
        <div class="opt"><h3>A</h3><audio id="audioA" controls preload="none"></audio>
          <button data-side="left">A sounds more natural</button></div>
        <div class="opt"><h3>B</h3><audio id="audioB" controls preload="none"></audio>
          <button data-side="right">B sounds more natural</button></div>
      </div>
      <div style="text-align:center;margin-top:12px">
        <button class="ghost" id="tie">They sound about the same</button>
      </div>
    </div>
  </div>

  <div id="done" class="hidden card">
    <h2 style="margin-top:0;font-size:17px">Done — thank you</h2>
    <p>Send the file below back to whoever asked you to run this.</p>
    <button class="primary" id="download">Download my responses</button>
    <div class="warn" id="dlwarn">If the download does not start (some browsers block it for
      local files), copy the text below into a file instead.</div>
    <textarea id="payload" readonly></textarea>
    <button class="ghost" id="copy" style="margin-top:10px">Copy to clipboard</button>
  </div>
</div>

<script>
const TRIALS = {trials_json};
const CLIPS = {clips_json};
const TEXT = {text_json};
const CATS = {cats_json};
const KEY = "smean-khmer-listening-v1";

let listener = "", i = 0, votes = [], startedAt = null;
const $ = id => document.getElementById(id);
const src = (model, id) => "data:audio/mpeg;base64," + CLIPS[model][id];

function save() {{
  try {{ localStorage.setItem(KEY, JSON.stringify({{listener, i, votes, startedAt}})); }}
  catch (e) {{ /* private mode, quota: the test still works, it just cannot resume */ }}
}}

function restore() {{
  try {{
    const raw = localStorage.getItem(KEY);
    if (!raw) return false;
    const s = JSON.parse(raw);
    if (!s || !Array.isArray(s.votes) || s.i >= TRIALS.length) return false;
    listener = s.listener || ""; i = s.i; votes = s.votes; startedAt = s.startedAt;
    return confirm("Resume where you left off (" + i + " of " + TRIALS.length + " done)?");
  }} catch (e) {{ return false; }}
}}

function render() {{
  if (i >= TRIALS.length) return finish();
  const t = TRIALS[i];
  $("sentence").textContent = TEXT[t.id];
  $("counter").textContent = "Trial " + (i + 1) + " of " + TRIALS.length;
  $("cat").textContent = CATS[t.id] || "";
  $("prog").style.width = (100 * i / TRIALS.length) + "%";
  const a = $("audioA"), b = $("audioB");
  a.pause(); b.pause();
  a.src = src(t.left, t.id); b.src = src(t.right, t.id);
  a.load(); b.load();
}}

function vote(choice) {{
  const t = TRIALS[i];
  votes.push({{
    id: t.id, kind: t.kind, choice: choice,
    left: t.left, right: t.right,
    picked: choice === "tie" ? "tie" : (choice === "left" ? t.left : t.right),
    expected: t.expected || null,
    at: new Date().toISOString()
  }});
  i++; save(); render();
}}

function finish() {{
  $("test").classList.add("hidden");
  $("done").classList.remove("hidden");
  const payload = JSON.stringify({{
    listener: listener, started: startedAt, finished: new Date().toISOString(),
    n_trials: TRIALS.length, votes: votes
  }}, null, 2);
  $("payload").value = payload;
  $("download").onclick = () => {{
    const blob = new Blob([payload], {{type: "application/json"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "listening-" + (listener || "anon").replace(/[^A-Za-z0-9]+/g, "_") + ".json";
    document.body.appendChild(a); a.click(); a.remove();
  }};
  $("copy").onclick = () => {{
    $("payload").select();
    try {{ document.execCommand("copy"); $("copy").textContent = "Copied"; }} catch (e) {{}}
  }};
  try {{ localStorage.removeItem(KEY); }} catch (e) {{}}
}}

$("start").onclick = () => {{
  listener = $("listener").value.trim();
  if (!listener) {{ $("listener").focus(); return; }}
  startedAt = startedAt || new Date().toISOString();
  $("intro").classList.add("hidden");
  $("test").classList.remove("hidden");
  save(); render();
}};
document.querySelectorAll("[data-side]").forEach(btn => {{
  btn.onclick = () => vote(btn.dataset.side);
}});
$("tie").onclick = () => vote("tie");

if (restore()) {{
  $("listener").value = listener;
  $("intro").classList.add("hidden");
  $("test").classList.remove("hidden");
  render();
}}
</script>
"""


def main():
    ap = argparse.ArgumentParser(description="Build a blind A/B naturalness test")
    ap.add_argument("--pair", default="voxcpm2:higgs3",
                    help="the two models to compare, colon-separated")
    ap.add_argument("--sentences", type=int, default=40,
                    help="40 x 5 listeners = 200 trials, enough to resolve a 60%% "
                         "preference; 30 only resolves 65%% and up")
    ap.add_argument("--anchors", type=int, default=6,
                    help="catch trials against a known-bad model. Six is the floor: "
                         "with four, a listener clicking at random passes a 75%% "
                         "threshold 31%% of the time")
    ap.add_argument("--bitrate", default="64k")
    ap.add_argument("--seed", type=int, default=0,
                    help="selects sentences and fixes the A/B assignment baked into the page")
    args = ap.parse_args()

    pair = tuple(args.pair.split(":"))
    if len(pair) != 2:
        raise SystemExit("--pair wants exactly two models, e.g. voxcpm2:higgs3")

    entries = pick_sentences(load_entries(), args.sentences, args.seed)
    trials = build_trials(entries, pair, args.anchors, args.seed)

    needed = {}
    for trial in trials:
        for model in (trial["left"], trial["right"]):
            needed.setdefault(model, set()).add(trial["id"])

    clips, total = {}, 0
    for model, ids in needed.items():
        clips[model] = {}
        for entry_id in sorted(ids):
            encoded = encode_mp3(wav_path(model, entry_id), args.bitrate)
            if encoded is None:
                raise SystemExit(f"missing or unencodable audio: {model}/{entry_id}.wav")
            clips[model][entry_id] = encoded
            total += len(encoded)
    if total > MAX_PAGE_BYTES:
        raise SystemExit(
            f"embedded audio is {total / 1e6:.1f} MB, over the {MAX_PAGE_BYTES / 1e6:.0f} MB "
            f"budget. Lower --sentences or --bitrate."
        )

    by_id = {e["id"]: e for e in load_entries()}
    text = {t["id"]: by_id[t["id"]]["sentence"] for t in trials}
    cats = {t["id"]: by_id[t["id"]].get("category", "") for t in trials}

    page = PAGE.format(
        trials_json=json.dumps(trials),
        clips_json=json.dumps(clips),
        text_json=json.dumps(text, ensure_ascii=False),
        cats_json=json.dumps(cats, ensure_ascii=False),
        minutes=max(3, round(len(trials) * 0.35)),
    )
    OUT_PATH.write_text(page, encoding="utf-8")

    # The key is written separately: it must not travel with the page a
    # listener opens, or the test stops being blind.
    key_path = ROOT / "evaluation" / "results" / "listening_test_key.json"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({
        "pair": list(pair), "seed": args.seed, "n_test": args.sentences,
        "n_anchor": args.anchors, "trials": trials,
    }, indent=2), encoding="utf-8")

    print(f"wrote {OUT_PATH.relative_to(ROOT)}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"      {len(trials)} trials -- {args.sentences} {pair[0]} vs {pair[1]}, "
          f"{args.anchors} anchor")
    print(f"      key: {key_path.relative_to(ROOT)}")
    print(f"\nSend the html to each listener. Collect their .json files, then:")
    print(f"      python evaluation/listening_analyse.py <files...>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
