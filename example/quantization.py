"""
# Install instructions

```sh
pip install -e .
python examples/quantization.py
```

# baseline without quantization:

generate: by using use_torch_compile=True, the first step would take long
generate step 86: speed=22.813 tokens/s, realtime factor=0.265x
generate step 172: speed=74.991 tokens/s, realtime factor=0.872x
generate step 258: speed=73.515 tokens/s, realtime factor=0.855x
generate step 344: speed=77.206 tokens/s, realtime factor=0.898x
generate step 430: speed=74.883 tokens/s, realtime factor=0.871x
generate: total step=486, total duration=10.107s

# with torchao quantization:

generate: by using use_torch_compile=True, the first step would take long
generate step 86: speed=3.616 tokens/s, realtime factor=0.042x
generate step 172: speed=49.577 tokens/s, realtime factor=0.576x
generate step 258: speed=49.666 tokens/s, realtime factor=0.578x
generate step 344: speed=50.187 tokens/s, realtime factor=0.584x
generate step 430: speed=50.133 tokens/s, realtime factor=0.583x
generate step 516: speed=48.332 tokens/s, realtime factor=0.562x
generate: total step=585, total duration=35.129s
"""
import os
import logging
import time

import torch
from dia.model import Dia
import numpy as np
from torch import nn

from torchao.quantization.quant_api import (
    quantize_,
    Int8DynamicActivationInt8WeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)
import torchao

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


logger.info("loading model")
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="bfloat16")

TEXT = """Completely reproducible results are not guaranteed across PyTorch
releases, individual commits, or different platforms."""

def generate(text,  seed=15, use_torch_compile=True):
    text = "[S2] " + text
    logger.info(f"generating {seed=} {text=}")

    torch.manual_seed(seed)
    output = model.generate(text, use_torch_compile=use_torch_compile, verbose=True)

    model.save_audio(f"simple.mp3", output)


model.model.to(torch.bfloat16)

quantize_(model.model, Int8DynamicActivationInt8WeightConfig())

generate(TEXT)
