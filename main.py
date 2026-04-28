"""
main.py — RPCA-based vocal separation (manual version)
"""

from pathlib import Path
import soundfile as sf

from pca_separation import run_rpca_full_manual

INPUT_FILE = "MIR-1k(small)/amy_3_01.wav"
OUTPUT_DIR = "output_files"

SR = 16_000

LAMBDA = 0.07
GAIN = 1.0
MASK_POWER = 2.0

def main():
    base = Path(__file__).parent
    input_path = (base / INPUT_FILE).resolve()
    output_dir = (base / OUTPUT_DIR).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_path = output_dir / f"{input_path.stem}_vocal.wav"
    inst_path = output_dir / f"{input_path.stem}_inst.wav"

    print(f"Input: {input_path.name}")
    print("Running RPCA separation...\n")
    x, sr = sf.read(str(input_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    voice, inst = run_rpca_full_manual(x)
    sf.write(str(voice_path), voice, sr)
    sf.write(str(inst_path), inst, sr)
    print("Done!")
    print(f"Vocal → {voice_path.name}")
    print(f"Inst  → {inst_path.name}")


if __name__ == "__main__":
    main()