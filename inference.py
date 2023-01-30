import argparse
import torch
import glob
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext

import pandas as pd
import pdb

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels

import pdb

model_name = "mn40_as_ext"
cuda = False
sample_rate = 32000
window_size = 800
hop_size = 320
n_mels = 128

parser = argparse.ArgumentParser(description='Example of parser. ')
# model name decides, which pre-trained model is loaded
parser.add_argument('--model_name', type=str, default='mn40_as_ext')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--audio_path', type=str, required=False, default=None)

# preprocessing
parser.add_argument('--sample_rate', type=int, default=32000)
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--hop_size', type=int, default=320)
parser.add_argument('--n_mels', type=int, default=128)

# overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
parser.add_argument('--ensemble', nargs='+', default=["mn40_as_ext", "mn40_as", "mn40_as_no_im_pre"])

args = parser.parse_args()

device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

"""
Running Inference on an audio clip.
"""
model_name = args.model_name
device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
audio_path = args.audio_path
sample_rate = args.sample_rate
window_size = args.window_size
hop_size = args.hop_size
n_mels = args.n_mels

# load pre-trained model
if len(args.ensemble) > 0:
    model = get_ensemble_model(args.ensemble)
else:
    model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
model.to(device)
model.eval()
# pdb.set_trace()
# model to preprocess waveform into mel spectrograms
mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
mel.to(device)
mel.eval()

def audio_tagging(args):
    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
            preds[sorted_indexes[k]]))
    print("********************************************************")

def inference(audio_path):
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]
    # print(labels)
    # print("preds", preds)
    # print("sorted_indexes", sorted_indexes)
    candidates = {"Vehicle horn": preds[308],
                  "Baby Crying": preds[23],
                  "Speech": preds[0],
                  "Whistle": preds[402],}
    # pdb.set_trace()


    result = {}
    result.update({"Vehicle horn": preds[308]})
    result.update({"Baby Crying": preds[32]})
    result.update({"Whistle": preds[402]})
    result.update({"Speech": preds[0]})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn40_as')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, required=True)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()

    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name)
    model.to(device)
    model.eval()
    
    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    audio_paths = glob.glob("resources/*.wav")
    print(audio_paths)

    df = pd.DataFrame(labels)

    print(df[0][300])

    #############################################################
    # test inference one time
    #############################################################
    print("\nCheck If model is loaded\n")
    audio_path = "resources/-3Y4NuyZmvA_250.wav"

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
            preds[sorted_indexes[k]]))
    print("********************************************************")
    print("\nModel loaded successfully\n")

    #############################################################
    # test inference for one time
    #############################################################
    print(inference(audio_path))

    
    # # testing if infrerence working
    # for audio_path in audio_paths:
    #     (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    #     waveform = torch.from_numpy(waveform[None, :]).to(device)

    #     # our models are trained in half precision mode (torch.float16)
    #     # run on cuda with torch.float16 to get the best performance
    #     # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    #     with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
    #         spec = mel(waveform)
    #         preds, features = model(spec.unsqueeze(0))
    #     preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    #     sorted_indexes = np.argsort(preds)[::-1]

    #     # Print audio tagging top probabilities
    #     print("************* Acoustic Event Detected: *****************")
    #     for k in range(10):
    #         print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
    #             preds[sorted_indexes[k]]))
    #     print("********************************************************")

    # audio_tagging(args)
    