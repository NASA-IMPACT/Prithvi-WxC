# Prithvi WxC: Foundation model for weather and climate

This repository contains the code of the Prithvi WxC foundation model as well as a basic zero-shot examples for testing and illustration. For fine-tuning applications please refer to task-specific repositories listed [below](https://github.com/NASA-IMPACT/Prithvi-WxC?tab=readme-ov-file#fine-tuning-applications).

## Architecture overview: A scalable and flexible vision transformer

Prithvi WxC, a scalable 2D vision transformer inspired by Hiera, overcomes architectural limitations to handle non-rectangular data topologies. It leverages a pretraining strategy with attention and fine-tuning with convolutions, drawing from both Hiera and MaxViT approaches.

Our data, structured into windows, takes the shape (batch, windows, tokens, features). We alternate between **local attention** (within a window) and **global attention** (across windows), akin to modulo masking. This is implemented by transposing dimensions between transformer layers. Attention acts on the third dimension, the second being part of the batch. Masking can target entire windows or individual tokens, the latter disrupting global connections between the same token across windows. See the figure for illustration:

![arch_main](https://github.com/user-attachments/assets/2a7eeb73-2ee4-485b-9756-83410866d09a)


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
   pip install '.[examples]'
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

Prithvi WxC used data from the MERRA-2 reanalysis for pretraining. Go to [GES DISC](https://disc.gsfc.nasa.gov/) to download the input variables (you may need to create an account there). Next, refer to the `data_preparation/prepare_data.py` script to create the input data for the model.

``` shell
python preproc.py -i /path/to/raw_input -s /path/to/static -o path/to/out_data -d "20241004"
```

Please, note that the structure of your `/path/to/raw_input` is expected to be:

```
 - /path/to/raw_input
 |- raw
   |- 2024
     |- MERRA2_400.inst1_2d_asm_Nx.20241003.nc4
     |- MERRA2_400.inst1_2d_asm_Nx.20241004.nc4
        ...
     |- MERRA2_400.tavg1_2d_rad_Nx.20241011.nc4
```

where `2024` folder contains `MERRA2_400.inst1_2d_asm_Nx.YYYYMMDD`, `MERRA2_400.inst3_3d_asm_Nv.YYYYMMDD`, `MERRA2_400.tavg1_2d_flx_Nx.YYYYMMDD`, `MERRA2_400.tavg1_2d_lnd_Nx.YYYYMMDD`, `MERRA2_400.tavg1_2d_rad_Nx.YYYYMMDD` files ***from the day before to the final date***. These files can be downloaded from GES DISC and you can find instruction on how to download them [HERE](https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Access%20GES%20DISC%20Data%20Using%20wget%20and%20curl). The datasets you need are, as described in the paper ([link](https://arxiv.org/html/2409.13598v1#bib)), the following:

 - Surface
   - M2I1NXASM
   - M2T1NXLND
   - M2T1NXFLX
   - M2T1NXRAD
 - Vertical
   - M2I3NVASM
 - Static
   - M2C0NXASM
   - M2CONXCTM

Static files are expected to reside in the root of `/path/to/static`.

The climatology is computed from MERRA-2 data, and it is [available via Hugging Face](https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1/tree/main/climatology). See the paper for details on variables choosen and the methodology behind the climatology.


## Citation
If you use this work, consider citing our paper

```
@misc{schmude2024prithviwxcfoundationmodel,
      title={Prithvi WxC: Foundation Model for Weather and Climate}, 
      author={Johannes Schmude and Sujit Roy and Will Trojak and Johannes Jakubik and Daniel Salles Civitarese and Shraddha Singh and Julian Kuehnert and Kumar Ankur and Aman Gupta and Christopher E Phillips and Romeo Kienzler and Daniela Szwarcman and Vishal Gaur and Rajat Shinde and Rohit Lal and Arlindo Da Silva and Jorge Luis Guevara Diaz and Anne Jones and Simon Pfreundschuh and Amy Lin and Aditi Sheshadri and Udaysankar Nair and Valentine Anantharaj and Hendrik Hamann and Campbell Watson and Manil Maskey and Tsengdar J Lee and Juan Bernabe Moreno and Rahul Ramachandran},
      year={2024},
      eprint={2409.13598},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.13598}, 
}
```
