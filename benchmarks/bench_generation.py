# coding=utf-8
# Copyright 2026 sakusaku-rich
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure where a warm generation spends its time.

The investigation notes list three things to confirm before optimizing further:
the breakdown of a warm run, whether the predictor CUDA Graph is applied, and
what a reused reference prompt costs. This script measures all three on the
target GPU and prints one block per stage.

What is measured per timed run (every run reuses the same voice clone prompt):

- End-to-end wall time of `generate_voice_clone`, bounded by `cuda.synchronize`.
- Every `Qwen3TTSTalkerModel` forward on the GPU timeline: the first call is the
  prefill, the rest are the decode frames.
- Every code predictor call (one per frame) on the GPU timeline, plus the
  offset of the first frame as a codec-level TTFA proxy.
- The speech decoder on the GPU timeline.
- The residual between the stage spans and the wall time, which is the CPU-side
  work that the GPU does not cover.
- Peak CUDA memory.

A stage span is the elapsed time between two CUDA events, so it includes the
gaps where the GPU waits for the CPU to launch the next kernel. Compare it with
the profiler's busy time to tell whether a stage is CPU bound.

`--profile` runs one extra generation under `torch.profiler` and prints the
kernel launch count, the GPU busy time and the top kernels.

Example:

    python benchmarks/bench_generation.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --label long --runs 10 --profile
"""

import argparse
import statistics
import time

import torch

from qwen_tts import Qwen3TTSModel

DEFAULT_REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
DEFAULT_REF_TEXT = ("Okay. Yeah. I resent you. I love you. I respect you. "
                    "But you know what? You blew it! And thanks to you.")
TEXTS = {
    "short": "今日は晴れている。",
    "long": "朝の電車で本を読みながら、週末の予定を考えた。駅に着いたら、まず友人に電話をかけよう。",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--label", choices=sorted(TEXTS), default="short",
                        help="input text to synthesize")
    parser.add_argument("--text", default=None, help="override the label text")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--x-vector-only", action="store_true",
                        help="skip the reference codec (ICL off)")
    parser.add_argument("--rebuild-prompt", action="store_true",
                        help="build the voice clone prompt on every run instead of "
                             "reusing one, to measure what the reuse saves")
    parser.add_argument("--runs", type=int, default=8, help="timed runs")
    parser.add_argument("--warmup", type=int, default=2,
                        help="untimed runs; the first one captures the CUDA Graph")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1234,
                        help="torch seed reset before every run")
    parser.add_argument("--attn", default="flash_attention_2",
                        help="attention implementation passed to from_pretrained")
    parser.add_argument("--profile", action="store_true",
                        help="profile one extra warm run with torch.profiler")
    return parser.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EventRecorder:
    """Wrap a bound method and time each call with CUDA events.

    The wrapper never synchronizes, so it does not change how the CPU runs
    ahead of the GPU. Read the spans after one final synchronize.
    """

    def __init__(self, obj, name):
        self.obj = obj
        self.name = name
        self.original = getattr(obj, name)
        self.spans = []  # (start_event, end_event) per call

    def __enter__(self):
        recorder = self

        def wrapper(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            result = recorder.original(*args, **kwargs)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            recorder.spans.append((start, end))
            return result

        setattr(self.obj, self.name, wrapper)
        return self

    def __exit__(self, *exc):
        try:
            delattr(self.obj, self.name)
        except AttributeError:
            setattr(self.obj, self.name, self.original)
        return False

    def elapsed_ms(self):
        """GPU-timeline duration of each call, in call order.

        Events recorded inside a CUDA Graph capture are not valid timeline
        markers, so a span whose events cannot be timed is dropped. The talker
        graph path replays captured steps, which is exactly that case.
        """
        durations = []
        for start, end in self.spans:
            try:
                durations.append(start.elapsed_time(end))
            except RuntimeError:
                continue
        return durations

    def end_offset_ms(self, origin):
        """GPU-timeline offset of each call end from an origin event."""
        offsets = []
        for _, end in self.spans:
            try:
                offsets.append(origin.elapsed_time(end))
            except RuntimeError:
                continue
        return offsets


def stats_ms(values):
    if not values:
        return "n/a"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return (f"median {statistics.median(values):8.2f} | "
            f"mean {statistics.fmean(values):8.2f} | p95 {p95:8.2f} | "
            f"min {ordered[0]:8.2f} | max {ordered[-1]:8.2f} | n={len(values)}")


def report_environment(tts, args):
    model = tts.model
    talker = model.talker
    print("=" * 78)
    print("Environment")
    print(f"  torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"gpu {torch.cuda.get_device_name(0)}")
    try:
        import triton
        print(f"  triton {triton.__version__}")
    except ImportError:
        print("  triton unavailable")
    print(f"  model {args.model}")
    codec_config = getattr(getattr(model.speech_tokenizer, "model", None), "config", None)
    print(f"  attn implementations: talker {getattr(talker.config, '_attn_implementation', '?')} | "
          f"predictor {getattr(talker.code_predictor.config, '_attn_implementation', '?')} | "
          f"codec {getattr(codec_config, '_attn_implementation', '?')}")
    print(f"  dtype {next(model.parameters()).dtype} | tf32 matmul "
          f"{torch.backends.cuda.matmul.allow_tf32}")


def report_graph(tts, capture_ms):
    predictor = tts.model.talker.code_predictor
    graph = getattr(predictor, "_graph_cache", None)
    if graph is None:
        print("  predictor CUDA Graph: NOT captured (check the is_supported conditions)")
        return
    projected = getattr(graph, "projected_embeddings", "n/a")
    if projected == "n/a":
        tables = "n/a (older graph_predictor without precomputed tables)"
    elif projected is None:
        tables = "identity projection, no tables"
    else:
        per_table_mib = (projected[0].numel() * projected[0].element_size() / 2**20)
        tables = (f"{len(projected)} projected tables, "
                  f"{per_table_mib:.1f} MiB each, "
                  f"{per_table_mib * len(projected):.0f} MiB total")
    step_embeddings = getattr(graph, "step_position_embeddings", None)
    rotary = ("n/a" if step_embeddings is None
              else f"{len(step_embeddings) + 1} precomputed rotary tables")
    print(f"  predictor CUDA Graph: captured | {tables} | {rotary}")
    if capture_ms:
        print(f"  first frame cost (capture + precompute) {capture_ms:.0f} ms")


def run_generation(tts, prompt_items, text, args):
    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )
    return tts.generate_voice_clone(
        text=text,
        language="Japanese",
        voice_clone_prompt=prompt_items,
        **kwargs,
    )


def measure(tts, prompt_items, text, args):
    """Run one instrumented generation and return its measurements."""
    model = tts.model
    talker = model.talker
    measurement = {}
    origin = torch.cuda.Event(enable_timing=True)

    with EventRecorder(talker.model, "forward") as talker_frames, \
            EventRecorder(talker.code_predictor, "generate") as predictor_frames, \
            EventRecorder(model.speech_tokenizer, "decode") as decode_stage:
        sync()
        origin.record()
        start = time.perf_counter()
        wavs, sr = run_generation(tts, prompt_items, text, args)
        sync()
        measurement["e2e"] = time.perf_counter() - start
        measurement["wav_seconds"] = len(wavs[0]) / sr

    # talker.model.forward: the first call is the prefill, the rest are frames
    frame_ms = talker_frames.elapsed_ms()
    measurement["prefill_ms"] = frame_ms[0] if frame_ms else float("nan")
    measurement["talker_frame_ms"] = frame_ms[1:]
    # one predictor call per decode frame
    predictor_ms = predictor_frames.elapsed_ms()
    measurement["predictor_ms"] = predictor_ms
    measurement["predictor_total_ms"] = sum(predictor_ms)
    measurement["predictor_max_ms"] = max(predictor_ms) if predictor_ms else float("nan")
    offsets = predictor_frames.end_offset_ms(origin)
    measurement["first_frame_offset_ms"] = offsets[0] if offsets else float("nan")
    measurement["decode_ms"] = sum(decode_stage.elapsed_ms())
    return measurement


def report_run(index, m, warmup=False):
    tag = "warmup" if warmup else "run"
    frames = len(m["talker_frame_ms"])
    prompt = f" | prompt {m['prompt_s']:.3f} s" if m.get("prompt_s") else ""
    print(f"{tag} {index:2d} | e2e {m['e2e']:7.3f} s | audio {m['wav_seconds']:5.2f} s | "
          f"RTF {m['e2e'] / max(m['wav_seconds'], 1e-9):5.2f} | frames {frames}{prompt}")
    print(f"          prefill {m['prefill_ms']:7.2f} ms | talker frames "
          f"{sum(m['talker_frame_ms']):7.1f} ms | predictor {m['predictor_total_ms']:7.1f} ms | "
          f"speech decode {m['decode_ms']:7.1f} ms | first frame {m['first_frame_offset_ms']:7.2f} ms")


def report_aggregate(runs):
    print("=" * 78)
    print(f"Aggregate over {len(runs)} timed warm runs")
    e2e = [m["e2e"] * 1000 for m in runs]
    print(f"  e2e wall       : {stats_ms(e2e)} ms")
    rtf = [m["e2e"] / max(m["wav_seconds"], 1e-9) for m in runs]
    print(f"  RTF            : median {statistics.median(rtf):.2f}")
    print(f"  talker prefill : {stats_ms([m['prefill_ms'] for m in runs])} ms")
    frames = [t for m in runs for t in m["talker_frame_ms"]]
    print(f"  talker frame   : {stats_ms(frames)} ms")
    predictor = [t for m in runs for t in m["predictor_ms"]]
    print(f"  predictor frame: {stats_ms(predictor)} ms")
    print(f"  speech decode  : {stats_ms([m['decode_ms'] for m in runs])} ms")
    offsets = [m["first_frame_offset_ms"] for m in runs]
    print(f"  first frame    : {stats_ms(offsets)} ms (codec TTFA proxy)")

    # Stage shares of the median run
    med = statistics.median
    e2e_med = med([m["e2e"] * 1000 for m in runs])
    prefill_med = med([m["prefill_ms"] for m in runs])
    frames_med = med([sum(m["talker_frame_ms"]) for m in runs])
    predictor_med = med([m["predictor_total_ms"] for m in runs])
    decode_med = med([m["decode_ms"] for m in runs])
    other = e2e_med - prefill_med - frames_med - predictor_med - decode_med
    print("  stage shares of the median run:")
    for name, value in (("talker prefill", prefill_med),
                       ("talker decode frames", frames_med),
                       ("  of which predictor", predictor_med),
                       ("speech decode", decode_med),
                       ("other (prompt embeds, HF loop, sampling, copies)", other)):
        print(f"    {name:47s} {value:8.1f} ms ({value / e2e_med * 100:5.1f}%)")


def profile_run(tts, prompt_items, text, args):
    print("=" * 78)
    print("torch.profiler on one warm run")
    from torch.profiler import ProfilerActivity, profile
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run_generation(tts, prompt_items, text, args)
        sync()
    table = prof.key_averages()
    cuda_rows = [row for row in table if row.self_device_time_total > 0]
    launches = sum(row.count for row in cuda_rows)
    busy = sum(row.self_device_time_total for row in cuda_rows) / 1000
    print(f"  kernel launches {launches} | GPU busy {busy:.1f} ms")
    print(table.table(sort_by="self_device_time_total", row_limit=15))


def build_prompt(tts, args):
    """Build the voice clone prompt and return it with its wall time."""
    sync()
    start = time.perf_counter()
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
    )
    sync()
    return prompt_items, time.perf_counter() - start


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: this benchmark measures the GPU paths")
    text = args.text if args.text else TEXTS[args.label]
    torch.backends.cuda.matmul.allow_tf32 = False

    print("Loading model ...")
    sync()
    start = time.perf_counter()
    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    sync()
    print(f"  load {time.perf_counter() - start:.1f} s")
    report_environment(tts, args)

    prompt_items, prompt_s = build_prompt(tts, args)
    mode = "x-vector only" if args.x_vector_only else "ICL"
    print(f"  create_voice_clone_prompt {prompt_s:.3f} s ({mode})"
          + (" - rebuilt on every run" if args.rebuild_prompt
             else " - reused by every run below"))

    capture_ms = None
    runs = []
    for i in range(args.warmup + args.runs):
        if args.seed is not None:
            torch.manual_seed(args.seed)
        if args.rebuild_prompt:
            prompt_items, prompt_s = build_prompt(tts, args)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        m = measure(tts, prompt_items, text, args)
        m["prompt_s"] = prompt_s if args.rebuild_prompt else 0.0
        if i == 0:
            # The first eligible frame captures the graph and precomputes the
            # fixed tables inside this call.
            capture_ms = m["predictor_max_ms"]
        report_run(i, m, warmup=i < args.warmup)
        if i >= args.warmup:
            runs.append(m)
    report_graph(tts, capture_ms)
    print(f"  peak CUDA memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    report_aggregate(runs)
    if args.profile:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        profile_run(tts, prompt_items, text, args)


if __name__ == "__main__":
    main()
