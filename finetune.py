import argparse
import torch
import torch.functional as F
import glob
import librosa
import numpy as np
from torch import nn
from torch import autocast
from contextlib import nullcontext
import torchinfo
import torchsummary
import os

import pandas as pd
import pdb

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels

labels_map = {
    0: "Vehicle horn",
    1: "Baby cry",
    2: "Glass clink",
    3: "",
}

dataset_classes = 527
output_classes = 4

parser = argparse.ArgumentParser(description='Example of parser. ')
# model name decides, which pre-trained model is loaded
parser.add_argument('--model_name', type=str, default='mn40_as_ext')
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--audio_path', type=str, required=False, default="resources/--2XRMjyizo_0.wav")

# training
parser.add_argument('--pretrained_name', type=str, default=None)
parser.add_argument('--model_width', type=float, default=1.0)
parser.add_argument('--head_type', type=str, default="mlp")
parser.add_argument('--se_dims', type=str, default="c")
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--mixup_alpha', type=float, default=0.3)
parser.add_argument('--epoch_len', type=int, default=100000)
parser.add_argument('--roll', action='store_true', default=False)
parser.add_argument('--wavmix', action='store_true', default=False)
parser.add_argument('--gain_augment', type=int, default=0)
parser.add_argument('--weight_decay', type=int, default=0.0001)

# lr schedule
parser.add_argument('--max_lr', type=float, default=0.0008)
parser.add_argument('--warm_up_len', type=int, default=8)
parser.add_argument('--ramp_down_start', type=int, default=80)
parser.add_argument('--ramp_down_len', type=int, default=95)
parser.add_argument('--last_lr_value', type=float, default=0.01)

# preprocessing
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--resample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--freqm', type=int, default=0)
parser.add_argument('--timem', type=int, default=0)
parser.add_argument('--fmin', type=int, default=0)
parser.add_argument('--fmax', type=int, default=None)

# overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as_ext"])
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as"])
# parser.add_argument('--ensemble', nargs='+', default=["mn40_as_no_im_pre"])
parser.add_argument('--ensemble', nargs='+', default=["mn10_as", "mn40_as_ext"])

args = parser.parse_args()

model_name = args.model_name
device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
print(f"{device=}")
audio_path = args.audio_path

sample_rate = args.sample_rate
window_size = args.window_size
hop_size = args.hop_size
n_mels = args.n_mels

args = parser.parse_args()

# fully connected layer 527 -> 527 -> 527 -> 4
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define layers
        layers = [nn.Linear(dataset_classes, dataset_classes), # 527 x 527
                  nn.ReLU(),
                  nn.Linear(dataset_classes, dataset_classes), # 527 x 527
                  nn.ReLU(),
                  nn.Linear(dataset_classes, output_classes), # 527 x 4
                  nn.Sigmoid()]
        # fully connected layer
        self.fc = nn.Sequential(*layers)
        
        # initialize model parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class Full_network(nn.Module):
    def __init__(self, args):
        super(Full_network, self).__init__()
        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=args.n_mels,
                            sr=args.resample_rate,
                            win_length=args.window_size,
                            hopsize=args.hop_size,
                            n_fft=args.n_fft,
                            freqm=args.freqm,
                            timem=args.timem,
                            fmin=args.fmin,
                            fmax=args.fmax
                            )
        
        self.mel.to(device)
        self.mel.train()

        # load pre-trained model
        if len(args.ensemble) > 0:
            self.model = get_ensemble_model(args.ensemble)
        else:
            self.model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
        self.model.to(device)
        self.model.train()

        # **freeze** model parameters for finetuning
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = Net()
        self.fc.to(device)
        self.fc.train()

    def forward(self, waveform):
        # (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        # waveform = torch.from_numpy(waveform[None, :]).to(device)
        with autocast(device_type=device.type) if args.cuda else nullcontext():
            spec = self.mel(waveform)
            preds, features = self.model(spec.unsqueeze(0))

        preds=preds.float() # change the torch.float16 to torch.float32
        output = self.fc(preds)
        return output
    
###############################################################
# debugging network 
temp = torch.randn(527, requires_grad=True)
network = Net()
print(network(temp).shape) # torch.size(4)
###############################################################

# network = Full_network(args)
# print(network(audio_path).shape)
# pdb.set_trace()
###############################################################

def train(args):
    network = Full_network(args)

    folder_list = glob.glob("../audioset-processing/output/*[!csv]")
    for folder in folder_list:
        pass
    print(f"Results on AudioSet test split for loaded model: {model_name}")
    print("  mAP: {:.3f}".format(mAP.mean()))
    print("  ROC: {:.3f}".format(ROC.mean()))

    # (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    # waveform = torch.from_numpy(waveform[None, :]).to(device)

    pass

    # device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # # model to preprocess waveform into mel spectrograms
    # mel = AugmentMelSTFT(n_mels=args.n_mels,
    #                      sr=args.resample_rate,
    #                      win_length=args.window_size,
    #                      hopsize=args.hop_size,
    #                      n_fft=args.n_fft,
    #                      freqm=args.freqm,
    #                      timem=args.timem,
    #                      fmin=args.fmin,
    #                      fmax=args.fmax
    #                      )
    
    # mel.to(device)

    # # load prediction model
    # pretrained_name = args.pretrained_name
    # if pretrained_name:
    #     model = get_mobilenet(width_mult=NAME_TO_WIDTH(pretrained_name), pretrained_name=pretrained_name,
    #                           head_type=args.head_type, se_dims=args.se_dims)
    # else:
    #     model = get_mobilenet(width_mult=args.model_width, head_type=args.head_type, se_dims=args.se_dims)
    # model.to(device)

    # # freeze parameters from original model
    # for param in model.parameters():
    #     param.requires_grad = False
    # temp = torch.randn((1,1,128,1000))
    # temp = temp.to(device)
    # pdb.set_trace()

    # # add final layer 
    # fc = Net() # fully connected layer
    # fc.to(device)
    # model = nn.Sequential(model, fc)

train(args)

# load dataset_dir
dataset_dir = None
assert dataset_dir is not None, "Specify AudioSet location in variable 'dataset_dir'. " \
                                "Check out the Readme file for further instructions. " \
                                "https://github.com/fschmid56/EfficientAT/blob/main/README.md"

dataset_config = {
    'balanced_train_hdf5': os.path.join(dataset_dir, "balanced_train_segments_mp3.hdf"),
    'unbalanced_train_hdf5': os.path.join(dataset_dir, "unbalanced_train_segments_mp3.hdf"),
    'eval_hdf5': os.path.join(dataset_dir, "eval_segments_mp3.hdf"),
    'num_of_classes': 527
}