"""How models/dinov2-small.xml and .bin were made: DINOv2-small (facebook/dinov2-small,
Apache-2.0) converted for OpenVINO, with the pooling finder/looks.py expects built in.
The tool itself needs neither torch nor transformers; only this script does, in an
environment of its own:

    py -3.12 -m venv convert-venv
    convert-venv\\Scripts\\pip install torch transformers openvino==2026.4.1
    convert-venv\\Scripts\\python scripts\\convert_dinov2.py

Input: a batch of 448 x 448 RGB pages, scaled to 0-1 and normalised with ImageNet's mean
and spread. Output: 768 numbers a page, the class token then the mean of the patches.
Converting again changes the numbers slightly: retrain afterwards (retrain.bat).
"""
from pathlib import Path

import openvino as ov
import torch
from transformers import AutoModel

OUT = Path(__file__).parent.parent / "models" / "dinov2-small.xml"


class Pooled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

    def forward(self, pixels):
        h = self.model(pixel_values=pixels).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)


if __name__ == "__main__":
    with torch.inference_mode():
        model = ov.convert_model(Pooled(), example_input=torch.zeros(1, 3, 448, 448), input=[(-1, 3, 448, 448)])
    ov.save_model(model, OUT)   # weights stored as FP16 (42 MB), run as FP32
    print("saved", OUT)
