# SeeOurSounds-ML Repo

## Abstract
We are team from GDSC-Yonsei Warning is all you need. Our application is for the heraing defect especially who raise kids.

In this repository, we perform audioset multi-label-classification using pre-trained models described in the paper [Efficient Large-Scale Audio Tagging 
Via Transformer-To-CNN Knowledge Distillation](https://arxiv.org/pdf/2211.04772.pdf) and original code from (https://github.com/fschmid56/EfficientAT)

We use light model that can be performed in mobile device

---------------------------

The codebase is developed with *Python 3.10.8*. After creating an environment install the requirements as
follows:

```
pip install -r requirements.txt
```

## Pre-Trained Models

**Pre-trained models are available in the Github Releases and are automatically downloaded from there.** 
Loading the pre-trained models is as easy as running the following code piece:

```
from models.MobileNetV3 import get_model
model = get_model(width_mult=1.0, pretrained_name="mn10_as")
```

The Table shows all models contained in this repository. The naming convention for our models is 
**mn_\<width_mult\>_\<dataset\>**. In this sense, *mn10_as* defines a MobileNetV3 with parameter *width_mult=1.0*, pre-trained on 
AudioSet.

All models available are pre-trained on ImageNet [9] by default (otherwise denoted as 'no_im_pre'), followed by training on AudioSet [4]. The results appear slightly better than those reported in the
paper. We provide the best models in this repository while the paper is showing averages over multiple runs.

| Model Name       | Config                                             | Params (Millions) | MACs (Billions) | Performance (mAP) |
|------------------|----------------------------------------------------|-------------------|-----------------|-------------------|
| mn04_as          | width_mult=0.4                                     | 0.983             | 0.11            | .432              |
| mn05_as          | width_mult=0.5                                     | 1.43              | 0.16            | .443              |
| mn10_as          | width_mult=1.0                                     | 4.88              | 0.54            | .471              |
| mn20_as          | width_mult=2.0                                     | 17.91             | 2.06            | .478              |
| mn30_as          | width_mult=3.0                                     | 39.09             | 4.55            | .482              |
| mn40_as          | width_mult=4.0                                     | 68.43             | 8.03            | .484              |
| mn40_as_ext      | width_mult=4.0,<br/>extended training (300 epochs) | 68.43             | 8.03            | .487              |
| mn40_as_no_im_pre| width_mult=4.0, no ImageNet pre-training           | 68.43             | 8.03            | .483              |
| mn10_as_hop_15   | width_mult=1.0                                     | 4.88              | 0.36            | .463              |
| mn10_as_hop_20   | width_mult=1.0                                     | 4.88              | 0.27            | .456              |
| mn10_as_hop_25   | width_mult=1.0                                     | 4.88              | 0.22            | .447              |
| mn10_as_mels_40  | width_mult=1.0                                     | 4.88              | 0.21            | .453              |
| mn10_as_mels_64  | width_mult=1.0                                     | 4.88              | 0.27            | .461              |
| mn10_as_mels_256 | width_mult=1.0                                     | 4.88              | 1.08            | .474              |
| Ensemble         | width_mult=4.0, 9 Models                           | 615.87            | 72.27           | .498              |

Ensemble denotes an ensemble of 9 different mn40 models (3x mn40_as, 3x mn40_as_ext, 3x mn40_as_no_im_pre). 

The Parameter and Computational complexity (number of multiply-accumulates) is calculated using the script [complexity.py](complexity.py). Note that the number of MACs calculated with our procedure is qualitatively as it counts only the dominant operations (linear layers, convolutional layers and attention layers for Transformers). 

The complexity statistics of a model can be obtained by running:

```
python complexity.py --model_name="mn10_as"
```

Which will result in the following output:

```
Model 'mn10_as' has 4.88 million parameters and inference of a single 10-seconds audio clip requires 0.54 billion multiply-accumulate operations.
```

Note that computational complexity strongly depends on the resolution of the spectrograms. Our default is 128 mel bands and a hop size of 10 ms.

## Inference

You can use one of the pre-trained models for inference on a an audio file using the 
[inference.py](inference.py) script.  

For example, use **mn10_as** to detect acoustic events at a metro station in paris:

```
python inference.py --cuda --model_name=mn10_as --audio_path="resources/metro_station-paris.wav"
```

This will result in the following output showing the 10 events detected with the highest probability:

```
************* Acoustic Event Detected: *****************
Train: 0.811
Rail transport: 0.649
Railroad car, train wagon: 0.630
Subway, metro, underground: 0.552
Vehicle: 0.328
Clickety-clack: 0.179
Speech: 0.061
Outside, urban or manmade: 0.029
Music: 0.026
Train wheels squealing: 0.021
********************************************************
```

You can also use an ensemble for perform inference, e.g.:

```
python inference.py --ensemble mn40_as_ext mn40_as mn40_as_no_im_pre --cuda --audio_path=resources/metro_station-paris.wav
```


**Important:** All models are trained with half precision (float16). If you run float32 inference on cpu,
you might notice a slight performance degradation.

## Train and Evaluate on AudioSet

The training and evaluation procedures are simplified as much as possible. The most difficult part is to get AudioSet[4]
itself as it has a total size of around 1.1 TB and it must be downloaded from YouTube. Follow the instructions in 
the [PaSST](https://github.com/kkoutini/PaSST/tree/main/audioset) repository to get AudioSet in the format we need
to run the code in this repository. You should end up with three files:
* ```balanced_train_segmenets_mp3.hdf```
* ```unbalanced_train_segmenets_mp3.hdf```
* ```eval_segmenets_mp3.hdf```

Specify the folder containing the three files above in ```dataset_dir``` in the [dataset file](datasets/audioset.py).

Training and evaluation on AudioSet is implemented in the file [ex_audioset.py](ex_audioset.py).
#### Evaluation

To evaluate a model on the AudioSet evaluation data, run the following command:

```
python ex_audioset.py --cuda --model_name="mn10_as"
```

Which will result in the following output:

```
Results on AudioSet test split for loaded model: mn10_as
  mAP: 0.471
  ROC: 0.980
```

#### Training

Logging is done using [Weights & Biases](https://wandb.ai/site). Create a free account to log your experiments. During training
the latest model will be saved to the directory [wandb](wandb).

To train a model on AudioSet, you can run for example the following command:
```
python ex_audioset.py --cuda --train --pretrained_name=mn10_im_pytorch --batch_size=60 --max_lr=0.0004
```

Checkout the results of this example configuration [here](https://wandb.ai/florians/EfficientAudioTagging/reports/Training-mn10_as-from-ImageNet-pre-trained-on-a-GeForce-RTX-2080-Ti--VmlldzozMDMwMTc4).

To train a tiny model (```model_width=0.1```) with Squeeze-and-Excitation [10] on the frequency dimension and a fully convolutional
classification head, run the following command:

```
python ex_audioset.py --cuda --train --batch_size=120 --model_width=0.1 --head_type=fully_convolutional --se_dims=f
```

Checkout the results of this example configuration [here](https://wandb.ai/florians/EfficientAudioTagging/reports/Train-Tiny-Model-width-0-1---VmlldzozMDMwMjkx).



## References

[1] https://github.com/fschmid56/EfficientAT
