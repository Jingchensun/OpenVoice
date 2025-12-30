#!/usr/bin/env python3
"""
Batch voice-clone each reference audio into multiple speaking styles (emotion/voice style)
while keeping the transcription content the same.

Inputs:
  - A directory like:
      dataset/english_free_speech/files_cut_by_sentences/
        01_M_native/*.wav
        02_M_nonNative/*.wav
        ...
    Where each audio filename (without extension) is also the transcription text.

Outputs:
  - ./1_different_stytle/ (note: folder name follows the user's requirement)
    with files like:
      1_different_stytle/01_M_native/a dog fell out __whispering.wav

This script follows demo_part1.ipynb's pipeline:
  BaseSpeakerTTS(text, style) -> ToneColorConverter.convert(src_se -> tgt_se)

IMPORTANT:
  The OpenVoice checkpoints are NOT included in this repository by default.
  You must provide paths to:
    - base speaker ckpt dir (config.json + checkpoint.pth + *se.pth)
    - converter ckpt dir (config.json + checkpoint.pth)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch

from openvoice.api import BaseSpeakerTTS, ToneColorConverter


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


@dataclass(frozen=True)
class CkptPaths:
    base_dir: Path
    converter_dir: Path
    base_config: Path
    base_ckpt: Path
    converter_config: Path
    converter_ckpt: Path
    en_default_se: Path
    en_style_se: Path


def _die(msg: str, code: int = 2) -> None:
    print(f"[audio_clone.py] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _require_file(p: Path, hint: Optional[str] = None) -> None:
    if not p.is_file():
        if hint:
            _die(f"找不到文件：{p}\n提示：{hint}")
        _die(f"找不到文件：{p}")


def _resolve_ckpts(base_dir: Path, converter_dir: Path) -> CkptPaths:
    base_config = base_dir / "config.json"
    base_ckpt = base_dir / "checkpoint.pth"
    converter_config = converter_dir / "config.json"
    converter_ckpt = converter_dir / "checkpoint.pth"

    # demo_part1.ipynb uses these:
    en_default_se = base_dir / "en_default_se.pth"
    en_style_se = base_dir / "en_style_se.pth"

    _require_file(
        base_config,
        hint="你需要下载 OpenVoice 的 base speaker checkpoints，并把 EN 目录传给 --base_dir（里面应包含 config.json / checkpoint.pth / en_default_se.pth 等）",
    )
    _require_file(base_ckpt)
    _require_file(
        converter_config,
        hint="你需要下载 OpenVoice 的 converter checkpoints，并把 converter 目录传给 --converter_dir（里面应包含 config.json / checkpoint.pth）",
    )
    _require_file(converter_ckpt)
    _require_file(
        en_default_se,
        hint="demo_part1.ipynb 依赖 en_default_se.pth；请确认该文件存在于 base speaker 目录中。",
    )
    _require_file(
        en_style_se,
        hint="demo_part1.ipynb 的风格控制依赖 en_style_se.pth；请确认该文件存在于 base speaker 目录中。",
    )

    return CkptPaths(
        base_dir=base_dir,
        converter_dir=converter_dir,
        base_config=base_config,
        base_ckpt=base_ckpt,
        converter_config=converter_config,
        converter_ckpt=converter_ckpt,
        en_default_se=en_default_se,
        en_style_se=en_style_se,
    )


def _iter_audio_files(input_dir: Path, exts: Tuple[str, ...] = (".wav", ".mp3", ".flac")) -> Iterable[Path]:
    # expected layout: input_dir/<speaker_group>/*.(wav|mp3|flac)
    for speaker_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        for f in sorted(speaker_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                yield f


def _safe_filename(name: str) -> str:
    # Keep spaces/apostrophes (Linux ok), but remove path separators and trailing dots.
    name = name.replace("/", "_").replace("\\", "_")
    name = name.rstrip(".")
    if not name:
        name = "unnamed"
    return name


def _load_se_for_style(style: str, ckpts: CkptPaths, device: str) -> torch.Tensor:
    # demo_part1.ipynb: default uses en_default_se, other styles use en_style_se
    se_path = ckpts.en_default_se if style == "default" else ckpts.en_style_se
    return torch.load(se_path, map_location=torch.device(device)).to(device)


def _extract_target_se_direct(
    reference_audio: Path,
    converter: ToneColorConverter,
) -> torch.Tensor:
    # Fast path: directly extract from the reference audio (no VAD splitting).
    # Works well for short clips like your sentence-level wavs.
    return converter.extract_se(str(reference_audio), se_save_path=None).to(converter.device)


def _extract_target_se_vad(
    reference_audio: Path,
    converter: ToneColorConverter,
    processed_dir: Path,
) -> torch.Tensor:
    # Slow but robust path: use se_extractor.get_se(..., vad=True)
    from openvoice import se_extractor

    processed_dir.mkdir(parents=True, exist_ok=True)
    se, _audio_name = se_extractor.get_se(
        str(reference_audio),
        converter,
        target_dir=str(processed_dir),
        vad=True,
    )
    return se.to(converter.device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch OpenVoice style cloning for sentence-level audios.")
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
        help="输出目录（会自动创建）。",
    )
    ap.add_argument(
        "--base_dir",
        type=str,
        default="checkpoints/base_speakers/EN",
        help="BaseSpeakerTTS checkpoint 目录（需包含 config.json/checkpoint.pth/en_default_se.pth/en_style_se.pth）。",
    )
    ap.add_argument(
        "--converter_dir",
        type=str,
        default="checkpoints/converter",
        help="ToneColorConverter checkpoint 目录（需包含 config.json/checkpoint.pth）。",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda:0 或 cpu。",
    )
    ap.add_argument(
        "--styles",
        type=str,
        default=",".join(DEFAULT_STYLES),
        help=f"要生成的风格列表，逗号分隔。默认：{','.join(DEFAULT_STYLES)}",
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="TTS 语速（会影响 base TTS 的 length_scale）。",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.3,
        help="ToneColorConverter 的 tau 参数。",
    )
    ap.add_argument(
        "--watermark_message",
        type=str,
        default="@MyShell",
        help="converter.convert 的 message（水印/标记文本）。",
    )
    ap.add_argument(
        "--use_vad",
        action="store_true",
        help="使用 se_extractor.get_se 的 VAD 分割方式提取 target_se（更慢但更鲁棒）。默认使用直接 extract_se。",
    )
    ap.add_argument(
        "--processed_dir",
        type=str,
        default="processed",
        help="当 --use_vad 时用于存放中间分割音频/SE 的目录。",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 个音频（0 表示全量）。",
    )
    args = ap.parse_args()

    # Friendly note when users run it inside this repo without downloading checkpoints.
    # (This repo often omits large weight files.)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    converter_dir = Path(args.converter_dir).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()

    if not input_dir.is_dir():
        _die(f"--input_dir 不是目录：{input_dir}")

    ckpts = _resolve_ckpts(base_dir=base_dir, converter_dir=converter_dir)

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        _die(f"你指定了 device={device}，但当前环境没有可用 CUDA。请改用 --device cpu 或配置 GPU。")

    # Initialize models
    base_speaker_tts = BaseSpeakerTTS(str(ckpts.base_config), device=device)
    base_speaker_tts.load_ckpt(str(ckpts.base_ckpt))

    tone_color_converter = ToneColorConverter(str(ckpts.converter_config), device=device)
    tone_color_converter.load_ckpt(str(ckpts.converter_ckpt))

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if not styles:
        _die("styles 为空，请通过 --styles 提供至少一个风格。")

    # Validate style names exist in base_speaker_tts.hps.speakers
    available_styles = set(getattr(base_speaker_tts.hps, "speakers", {}).keys())
    missing = [s for s in styles if s not in available_styles]
    if missing:
        _die(
            "以下 styles 不在 base speaker 模型的 speakers 列表中："
            f"{missing}\n可用列表：{sorted(list(available_styles))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # tmp wav path for base tts output
    # Use a dedicated temp dir to avoid weird filenames and allow parallel-safe file creation later.
    tmp_root = Path(tempfile.mkdtemp(prefix="openvoice_audio_clone_"))

    count = 0
    for audio_path in _iter_audio_files(input_dir):
        speaker_group = audio_path.parent.name
        transcript = audio_path.stem  # filename without extension (keeps trailing spaces in the actual filename)

        # Target SE (clone this voice)
        if args.use_vad:
            target_se = _extract_target_se_vad(audio_path, tone_color_converter, processed_dir=processed_dir / speaker_group)
        else:
            target_se = _extract_target_se_direct(audio_path, tone_color_converter)

        group_out_dir = output_dir / speaker_group
        group_out_dir.mkdir(parents=True, exist_ok=True)

        for style in styles:
            source_se = _load_se_for_style(style, ckpts=ckpts, device=device)

            # Base TTS
            tmp_wav = tmp_root / "tmp.wav"
            base_speaker_tts.tts(
                transcript,
                str(tmp_wav),
                speaker=style,
                language="English",
                speed=float(args.speed),
            )

            out_name = _safe_filename(f"{audio_path.stem}__{style}.wav")
            out_path = group_out_dir / out_name

            # Convert tone color to target speaker
            tone_color_converter.convert(
                audio_src_path=str(tmp_wav),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(out_path),
                tau=float(args.tau),
                message=str(args.watermark_message),
            )

        count += 1
        if count % 10 == 0:
            print(f"[audio_clone.py] 已处理 {count} 个音频… 输出目录：{output_dir}")

        if args.limit and count >= args.limit:
            break

    print(f"[audio_clone.py] 完成：共处理 {count} 个音频。输出目录：{output_dir}")


if __name__ == "__main__":
    main()


