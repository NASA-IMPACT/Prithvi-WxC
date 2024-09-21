# Prithvi-WxC: Foundation model for weather and climate

This repository contains the code of the Prithvi WxC foundation model as well as a basic zero-shot examples for testing and illustration. For fine-tuning applications please refer to task-specific repositories

## Fine-tuning applications

We have fine-tuned the model to a number of downstream tasks. See the paper as well as the respective repository for details.

| Application                  | Dataset     | Repository                         |
| ---------------------------- | ----------- | ---------------------------------- |
| Downscaling                  | MERRA-2     | https://github.com/IBM/granite-wxc |
|                              | EURO-CORDEX | https://github.com/IBM/granite-wxc |
| Gravity wave parametrization | ERA5        | https://github.com/NASA-IMPACT/gravity-wave-finetuning |

Beyond these there are zero-shot applications in masked reconstruction and forecasting.

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

## Pretrained models

Prithvi WxC is a very flexible model. It has been pretrained on a pretext task blending masked reconstruction and forecasting so that it can be used for both zero-hours ahead as well as forecasting applications. Moreover, the masking pattern makes it suitable for both global and regional applications. There are currently two pretrained base models as well as several [fine-tuning applications](https://github.com/NASA-IMPACT/Prithvi-WxC?tab=readme-ov-file#fine-tuning-applications).

| Model                        | Details                                                                                                   | Weights                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| prithvi.wxc.2300m.v1         | Pretrained 2.3B parameter model. Flexible input and lead time. For general and 0-hour ahead applications. | https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1 |
| prithvi.wxc.rollout.2300m.v1 | Pretrained 2.3B parameter model. Input and lead time fixed to 6h. For forecasting applications.           | https://huggingface.co/Prithvi-WxC/prithvi.wxc.rollout.2300m.v1 |


## Data

Prithvi WxC used data from the MERRA-2 reanalysis for pretraining. In particular, the model uses a climatology computed from MERRA-2 data. The climatology, too, is [available via Hugging Face](https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1/tree/main/climatology). See the paper for details on variables choosen and the methodology behind the climatology.
