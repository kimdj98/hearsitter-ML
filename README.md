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

Change the audio_path to your own wav file path.

```
python inference.py --cuda --model_name=mn10_as --audio_path="resources/-fz6omiAhZ8_40.wav"
```

This will result in the following output showing the events detected that we chose to classify:

```
{'Car horn': 0.16050603486364706, 
 'Infant Crying': 0.000295591248141136, 
 'Glass': 0.0001893545668281149,
 'Fire alarm': 0.005741941637825221,
 'Gunshot': 0.00030780517881794366}
```

You can also use an ensemble for perform inference, e.g.:

```
python inference.py --ensemble mn40_as_ext mn40_as mn40_as_no_im_pre --cuda --audio_path="resources/-fz6omiAhZ8_40.wav"

```

**Important:** All models are trained with half precision (float16). If you run float32 inference on cpu,
you might notice a slight performance degradation.

------------------------------
Inside inference.py we made function **inference(audio_path)** for server purpose.

Inside inference.py change model in ensemble argument's default part based on pretrained-models above **parser.add_argument('--ensemble', nargs='+', defuault=["mn40_as", mn40_as_ext"])** if you don't want to use ensemble method just put an empty list []. and change model_name argument.

## References

[1] https://github.com/fschmid56/EfficientAT


Team github pages

hearsitter-server: 

(1) https://github.com/jimmy0006/hearsitter-server-python

(2) https://github.com/jimmy0006/hearsitter-server-go

hearsitter-flutter:

(1) https://github.com/gdsc-ys/hearsitter-flutter
