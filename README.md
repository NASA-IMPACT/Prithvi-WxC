# Prithvi-WxC: A foundation model for weather and climate

This repository contains the code of the Prithvi WxC foundation model as well as a basic zero-shot examples for testing and illustration. For fine-tuning applications please refer to task-specific repositories

## Getting started

1. Create a virtual environment
2. Clone this repository and install Prithvi WxC as a module
   ```
   git clone https://github.com/NASA-IMPACT/Prithvi-WxC
   cd Prithvi-WxC
   pip install .examples
   ```
3. Run one of the notebooks in the `examples` directory:
   - [Basic inference](examples/PrithviWxC_inference.ipynb)
   - [Rollout inference](examples/PrithviWxC_rollout.ipynb)
   
   These notebooks will download model weights as well as sample data for basic illustration from [Hugging Face](https://huggingface.co/Prithvi-WxC).

## Fine-tuning applications

| Application                  | Dataset     | Repository                         |
| ---------------------------- | ----------- | ---------------------------------- |
| Downscaling                  | MERRA-2     | https://github.com/IBM/granite-wxc |
|                              | EURO-CORDEX | https://github.com/IBM/granite-wxc |
| Gravity wave parametrization | ERA5        | https://github.com/NASA-IMPACT/gravity-wave-finetuning |

## Data

Prithvi WxC used data from the MERRA-2 reanalysis for pretraining. In particular, the model uses a climatology computed from MERRA-2 data. The climatology, too, is [available via Hugging Face](https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1/tree/main/climatology). See the paper for details on variables choosen and the methodology behind the climatology.
