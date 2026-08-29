#!/usr/bin/env bash
set -euo pipefail

if command -v tesseract >/dev/null 2>&1; then
  TESSERACT_BIN=(tesseract)
elif command -v conda >/dev/null 2>&1; then
  TESSERACT_BIN=(conda run -n jax-foam tesseract)
else
  echo "tesseract CLI not found" >&2
  exit 127
fi
for component in tes1_diffusion tes2_manufacture tes3_mesher tes4_fem; do
  "${TESSERACT_BIN[@]}" build "components/tesseracts/${component}"
done
