# Reverse Prompt - Hard Fork

From a given image, we first optimize a hard prompt using the algorithms and CLIP encoders. Then, we take the optimized prompts and feed them into Stable Diffusion to generate new images.

Colab Notebook for early testing [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fRXnkBEEgv4rPvVmesSU9BoLJEf-6Xrv?usp=sharing)
More Jupyter notebook examples can be found in the `examples/` folder.

## Dependencies

- PyTorch => 1.13.0
- transformers >= 4.23.1
- diffusers >= 0.11.1
- sentence-transformers >= 2.2.2
- ftfy >= 6.1.1
- mediapy >= 1.1.2

## Usage

```python
import open_clip
from optim_utils import *
import argparse
from PIL import Image

# load the target image
image = Image.open(image_path)

# load args
args = argparse.Namespace()
args.__dict__.update(read_json("sample_config.json"))

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

# You may modify the hyperparamters
args.prompt_len = 8 # number of tokens for the learned prompt

# optimize prompt
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=[image])
print(learned_prompt)
```

Note: \
`prompt_len`: number of tokens in the optimized prompt \
`batch_size`: number of target images/prompts been used for each iteraion \
`prompt_bs`: number of intializations

## Langugae Model Prompt Experiments

Check out `prompt_lm/` folder.

### TODOs

- Figure out how we can load multiple base images at the same time to comapre cosims
- Get 1.5 base working 512x512
