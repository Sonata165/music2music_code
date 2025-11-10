# Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization

This is the code for experiments in the paper [*Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization*](https://arxiv.org/abs/2408.15176). 

[ [Demo](https://www.oulongshen.xyz/automatic_arrangement) | [Paper](https://arxiv.org/abs/2408.15176) | [REMI-z MIDI Tookit](https://github.com/Sonata165/REMI-z) | [Author](https://www.oulongshen.xyz/) ]

## Structure

    .
    ├── baselines               Baseline implementations
    ├── dataset_preparation     Prepare dataset for training
    ├── evaluations             Functions to evaluate model performance
    ├── m2m                     Model training
    ├── tests                   Sanity check scripts
    ├── utils_arrange           Utility functions for arrangement inference
    ├── utils_chord             Utility functions for chord detection
    ├── utils_common            Common Utility functions
    ├── utils_instrument        Utility functions for instrument
    ├── utils_midi              Utility functions for MIDI, including the proposed tokenization
    └── utils_texture           Utility functions for texture analysis

Most useful scripts are located in the `m2m` directory, containing: 
- Model definition: `m2m_models.py` and `lightning_model.py`.
- Dataset class: `lightning_dataset.py`.
- Pre-train: `pretrain.py`
- Fine-tuning script: `lightning_train.py`
- Testing script: `lightning_test.py`
- Inference (band/piano arrangement): `reinst.py`
- Inference (drum arrangement): `drum_arrange.py`

The hyperparameter of all experiments are saved in `m2m/hparams`.


## Prepare Environment
    # Environment
    conda create -n m2m python=3.8
    conda activate m2m

    # Install PyTorch
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

    # Dependencies
    pip install -r requirements.txt   
    
    git clone https://github.com/Sonata165/REMI-z.git
    cd REMI-z
    pip install -r Requirements.txt
    pip install -e .


## Dataset preparation

    cd dataset_preparation

    # For pre-training
    python LAMD_to_remi.py

    # For fine-tuning
    python segment_data.py

## Training and Testing
Use below command for pre-train your own symbolic music generator: 

    python pretrain.py
Use below command to execute fine-tuning:
    
    python lightning_train.py [path/to/hparam]
    
    e.g.,
    python lightning_train.py hparams/elaborator_direct.yaml # our band arrangement model in the paper

After training, use below command for testing to get objective metrics (for voice separation and probing experiments)

    python lightning_test.py [path/to/hparam]

In particular, to obtain the results in the paper

    cd m2m

    # Generative pre-train
    python pretrain.py

    # Non-creative band arrangement for objective evaluation
    python lightning_train.py hparams/band_obj/ours.yaml

    # Band arrangement model
    python lightning_train.py hparams/band_arrange/elaborator.yaml

    # Piano reduction model
    python lightning_train.py hparams/piano_reduction/reduction_dur.yaml

    # Drum arrangement model
    python lightning_train.py hparams/drum_arrange/direct_opd.yaml

## Inference
To inference with models trained by yourself, use below command to arrange existing songs with new instrument set:

    python reinst.py [path/to/hparam]

Use below command to arrange a drum track for a song:

    python drum_arrange.py [path/to/hparam]

To directly use our pretrained model for inference, see `tutorial.ipynb` for details.

## Models

Here are models on huggingface:
|Model|Link|
|----|----|
|Pre-trained unconditional generation|[LongshenOu/m2m_pt](https://huggingface.co/LongshenOu/m2m_pt)|
|Band Arrangement Finetune | [LongshenOu/m2m_arranger](https://huggingface.co/LongshenOu/m2m_arranger) |
|Piano Reduction Finetune | [LongshenOu/m2m_pianist_dur](https://huggingface.co/LongshenOu/m2m_pianist_dur) |
|Drum Arrangement Finetune | [LongshenOu/m2m_drummer](https://huggingface.co/LongshenOu/m2m_drummer) |


See usage in `tutorial.ipynb`.


## Mapping from token to represented value
As in the appendix, you can find the mapping from token to their corresponding representing values, for time signature and tempo tokens, in `ts_dict.yaml` and `tempo_dict.yaml`, under the `utils_midi` directory.

## Citation

    @inproceedings{ou2025unifying,
        title     = {Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization},
        author    = {Ou, Longshen and Zhao, Jingwei and Wang, Ziyu and Xia, Gus and Liang, Qihao and Hopkins, Torin and Wang, Ye},
        booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
        year      = {2025}
    }