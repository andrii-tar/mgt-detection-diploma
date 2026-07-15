### solution to fix cuda on NVIDIA GeForce RTX 5060 Ti

uv pip install --reinstall --pre --index-strategy unsafe-best-match \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
