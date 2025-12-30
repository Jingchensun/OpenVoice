#!/usr/bin/env python3
"""
Compute Whisper-audio-embedding cosine similarity between:
  - input audios (original reference)
  - output audios (style-cloned by audio_clone.py)

For each input audio X and each style Y, audio_clone.py generates:
  output_dir/<speaker_group>/<safe_filename(f"{X.stem}__{Y}.wav")>

This script reproduces that mapping, extracts Whisper encoder embeddings for both
audios, computes cosine similarity, and saves lines to similarity.txt:
  input_audio_name<TAB>output_audio_name<TAB>similarity_score

Notes:
  - Requires: openai-whisper (import whisper) + ffmpeg available on PATH.
  - Embedding is mean-pooled over encoder time frames and L2-normalized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


DEFAULT_STYLES = [
    "default",
    "friendly",
    "cheerful",
    "excited",
    "sad",
    "angry",
    "terrified",
    "shouting",
    "whispering",
]


def _die(msg: str, code: int = 2) -> None:
    print(f"[similirity_compare.py] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _safe_filename(name: str) -> str:
    # Keep spaces/apostrophes (Linux ok), but remove path separators and trailing dots.
    name = name.replace("/", "_").replace("\\", "_")
    name = name.rstrip(".")
    if not name:
        name = "unnamed"
    return name


def _iter_audio_files(input_dir: Path, exts: Tuple[str, ...] = (".wav", ".mp3", ".flac")) -> Iterable[Path]:
    for speaker_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        for f in sorted(speaker_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                yield f


@torch.inference_mode()
def _whisper_embedding(
    audio_path: Path,
    model,
    device: str,
) -> torch.Tensor:
    """
    Returns a 1D, L2-normalized embedding on CPU.
    """
    try:
        import whisper  # type: ignore
    except Exception as e:
        _die(
            f"无法 import whisper（openai-whisper）。请在 openvoice 环境里执行：pip install openai-whisper。\n原始错误：{type(e).__name__}: {e}"
        )

    # Load and preprocess audio -> log-mel
    try:
        audio = whisper.load_audio(str(audio_path))
    except Exception as e:
        _die(
            f"读取音频失败：{audio_path}\n"
            f"请确认系统已安装 ffmpeg 且在 PATH 中（例如 `ffmpeg -version` 可运行）。\n"
            f"原始错误：{type(e).__name__}: {e}"
        )

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)  # (80, 3000)
    mel = mel.unsqueeze(0)  # (1, 80, 3000)

    # Match model dtype/device (whisper uses fp16 on cuda by default)
    model_param = next(model.parameters())
    mel = mel.to(device=device, dtype=model_param.dtype)

    # Encoder output: (1, T, D)
    enc = model.embed_audio(mel)
    vec = enc.mean(dim=1).squeeze(0)  # (D,)
    vec = torch.nn.functional.normalize(vec, dim=0)
    return vec.detach().to("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Whisper embedding cosine similarity for input vs output audios.")
    ap.add_argument(
        "--input_dir",
        type=str,
        default="dataset/english_free_speech/files_cut_by_sentences",
        help="输入音频目录（包含 01_M_native/ 这类子目录）。",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="1_different_stytle",
        help="输出音频目录（audio_clone.py 生成的目录）。",
    )
    ap.add_argument(
        "--styles",
        type=str,
        default=",".join(DEFAULT_STYLES),
        help=f"要对比的风格列表，逗号分隔。默认：{','.join(DEFAULT_STYLES)}",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper 模型大小，例如 tiny/base/small/medium/large-v3。",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda 或 cpu。",
    )
    ap.add_argument(
        "--output_txt",
        type=str,
        default="similarity.txt",
        help="输出文本文件路径（默认 similarity.txt）。",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 个 input 音频（0 表示全量）。",
    )
    ap.add_argument(
        "--skip_missing_output",
        action="store_true",
        help="如果某个 output 音频不存在则跳过（默认会跳过并打印 warning）。",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_txt = Path(args.output_txt).expanduser().resolve()

    if not input_dir.is_dir():
        _die(f"--input_dir 不是目录：{input_dir}")
    if not output_dir.is_dir():
        _die(f"--output_dir 不是目录：{output_dir}（请先运行 audio_clone.py 生成输出）")

    styles: List[str] = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles:
        _die("--styles 为空，请提供至少一个风格。")

    # Load whisper model
    try:
        import whisper  # type: ignore
    except Exception as e:
        _die(
            f"无法 import whisper（openai-whisper）。请在 openvoice 环境里执行：pip install openai-whisper。\n原始错误：{type(e).__name__}: {e}"
        )

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        _die(f"你指定了 device={device}，但当前环境没有可用 CUDA。请改用 --device cpu 或配置 GPU。")

    model = whisper.load_model(args.model, device=device)
    model.eval()

    emb_cache: Dict[str, torch.Tensor] = {}

    def get_emb(p: Path) -> torch.Tensor:
        key = str(p)
        if key not in emb_cache:
            emb_cache[key] = _whisper_embedding(p, model=model, device=device)
        return emb_cache[key]

    # Write results
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    written = 0
    missing_outputs = 0

    with output_txt.open("w", encoding="utf-8") as f_out:
        # 写入 CSV 表头
        f_out.write("input_audio,output_audio,similarity_score\n")
        
        for in_audio in _iter_audio_files(input_dir):
            speaker_group = in_audio.parent.name
            in_rel = in_audio.relative_to(input_dir)

            in_emb = get_emb(in_audio)

            for style in styles:
                out_name = _safe_filename(f"{in_audio.stem}__{style}.wav")
                out_audio = output_dir / speaker_group / out_name
                out_rel = out_audio.relative_to(output_dir)

                if not out_audio.is_file():
                    missing_outputs += 1
                    msg = f"[similirity_compare.py] WARNING: 找不到 output 音频，跳过：{out_audio}"
                    if args.skip_missing_output:
                        print(msg)
                    else:
                        print(msg)
                    continue

                out_emb = get_emb(out_audio)
                score = float(torch.sum(in_emb * out_emb).clamp(-1.0, 1.0).item())

                f_out.write(f"{in_rel},{out_rel},{score:.6f}\n")
                written += 1

            processed += 1
            if processed % 10 == 0:
                print(
                    f"[similirity_compare.py] 已处理 {processed} 个 input，写入 {written} 行。当前输出：{output_txt}"
                )

            if args.limit and processed >= args.limit:
                break

    print(
        f"[similirity_compare.py] 完成：input={processed}，写入行数={written}，缺失output={missing_outputs}。输出：{output_txt}"
    )


if __name__ == "__main__":
    main()


