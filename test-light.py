#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:12:43 2023

@author: carvalho

Usage:
    test.py [-h | --help]
    test.py [--version]
    test.py [--inputrep REP] [--gpu] [--mps] [--path MP] [--dpath DP] [--o OUT] [--nbframe NBFRAME]
            [--start START] [--end END] [--nbpoints POINTS] [--name NAME] [--runname NAME]

Options:
    -h --help  Show this helper
    --version  Show version and exit
    --gpu  Use GPU
    --mps  Use MPS
    --inputrep REP  Set the representation which will be used as input [default: signallike]
    --path MP  The path of the trained model [default: None]
    --dpath DP  The path of the MIDI files folder [default: None]
    --o OUT  Path of the output directory [default: None]
    --nbframe NBFRAME  Number of frames per bar [default: 16]
    --start START  Path of the starting bar [default: None]
    --end END  Path of the ending bar [default: None]
    --nbpoints POINTS  Number of points in the interpolation [default: 24]
    --name NAME  Name of the final MIDI files [default: None]
    --runname NAME  Name of the run [default: None]
"""

import os
import random
import time

import lightning as L
import models as m
import numpy as np
import pypianoroll
import representations as rep_classes
import torch
from docopt import docopt
import matplotlib.pyplot as plt


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if __name__ == '__main__':

    arguments = docopt(__doc__, version='symbolic_embeddings v1.0')
    print(arguments)

    # Set detect anomaly
    torch.autograd.set_detect_anomaly(True)  # type: ignore

    # Parameters
    train_path = arguments['--path'] + '/train'
    test_path = arguments['--path'] + '/test'
    # batch_size = int(arguments['--bsize'])
    nb_frame = int(arguments['--nbframe'])
    if arguments['--o'] == 'None':
        output_dr = os.getcwd() + '/output'
    else:
        output_dr = arguments['--o']

    # load the dataset
    if arguments['--inputrep'] == "pianoroll":
        # dataset = rep_classes.Pianoroll(train_path, nbframe_per_bar=nb_frame)
        testset = rep_classes.Pianoroll(test_path, nbframe_per_bar=nb_frame)
        input_dim = 128
        seq_length = nb_frame
    elif arguments['--inputrep'] == "midilike":
        # dataset = rep_classes.Midilike(train_path)
        testset = rep_classes.Midilike(test_path)
        input_dim = 1
    elif arguments['--inputrep'] == "midimono":
        # dataset = rep_classes.Midimono(train_path)
        testset = rep_classes.Midimono(test_path)
        input_dim = 1
    elif arguments['--inputrep'] == "signallike":
        # dataset = rep_classes.Signallike(
        #     train_path, nbframe_per_bar=nb_frame*2, mono=True)
        testset = rep_classes.Signallike(
            test_path, nbframe_per_bar=nb_frame*2, mono=True)
        input_dim = testset.signal_size//64
    elif arguments['--inputrep'] == "notetuple":
        # dataset = rep_classes.Notetuple(train_path)
        testset = rep_classes.Notetuple(test_path)
        input_dim = 5
    elif arguments['--inputrep'] == "dft128":
        # dataset = rep_classes.DFT128(train_path, nbframe_per_bar=nb_frame)
        testset = rep_classes.DFT128(test_path, nbframe_per_bar=nb_frame)
        input_dim = 130
        seq_length = nb_frame
    else:
        raise NotImplementedError(
            "Representation {} not implemented".format(arguments['--inputrep']))

    # Model parameters
    enc_hidden_size = 1024
    cond_hidden_size = 1024
    dec_hidden_size = 1024
    cond_outdim = 512
    num_layers_enc = 2
    num_layers_dec = 2
    num_subsequences = 4
    latent_size = 256

    if arguments['--inputrep'] in ['pianoroll', 'signallike', 'dft128']:
        output_dim = input_dim
    elif arguments['--inputrep'] == "midilike":
        output_dim = len(testset.vocabulary)  # type: ignore
        seq_length = 64
    elif arguments['--inputrep'] == "midimono":
        output_dim = 130
        seq_length = 16
    elif arguments['--inputrep'] == "notetuple":
        output_dim = sum([len(v) for v in
                          testset.vocabs]) + 129  # type: ignore
        seq_length = 32

    device = 'cpu'
    if arguments['--gpu'] and torch.cuda.is_available():  # type: ignore
        device = 'cuda'
    elif arguments['--mps'] and torch.backends.mps.is_available() and torch.backends.mps.is_built():  # type: ignore
        device = 'mps'

    # Instanciate model
    encoder = m.Encoder_RNN(input_dim, enc_hidden_size,
                            latent_size, num_layers_enc, device=device)
    decoder = m.Decoder_RNN_hierarchical(output_dim, latent_size, cond_hidden_size,  # type: ignore
                                         cond_outdim, dec_hidden_size=dec_hidden_size, num_layers=num_layers_dec,
                                         num_subsequences=num_subsequences, seq_length=seq_length)  # type: ignore

    if arguments['--inputrep'] == "notetuple":
        model = m.LightningVAE(encoder, decoder, arguments['--inputrep'],
                               vocab=testset.vocabs)  # type: ignore
    else:
        model = m.LightningVAE(encoder, decoder, arguments['--inputrep'])

    # Load the model
    accel = 'cpu'
    if arguments['--gpu']:
        accel = 'cuda'
    elif arguments['--mps']:
        accel = 'mps'

    last_model = f'{output_dr}/{arguments["--runname"]}/models/last.ckpt'
    last_checkpoint = torch.load(
        last_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(last_checkpoint["state_dict"])

    model.eval()

    testset.barfiles.sort()

    # For interpolations
    start_point = random.randint(0, len(testset))
    starting_point = testset[start_point]

    print("Starting point: ", start_point)

    end_point = random.randint(0, len(testset))
    ending_point = testset[end_point]

    latents = model.interpolate_from_points(starting_point, ending_point, 24)

    # Generate all the bars and concatenate them
    for i, latent in enumerate(latents):
        generated_bar = model.generate(latent)

        # clean the bar
        if arguments['--inputrep'] == 'signallike':
            pr_rec = testset.back_to_pianoroll(
                generated_bar.squeeze(0).flatten().detach().numpy())
            pr_rec[pr_rec <= 0.25] = 0
            pr_rec[pr_rec > 0.25] = 64
            y = pr_rec[:, ::2]
            for j in [0, 4, 8, 12]:
                y[:, j] = y[:, j+1]
            generated_bar = y.transpose(1, 0)
        if arguments['--inputrep'] == 'dft128':
            pr_rec = testset.back_to_pianoroll(
                generated_bar.squeeze(0).detach())

            #pr_rec[pr_rec < 0.8] = 0
            #pr_rec[pr_rec >= 0.8] = 1

            print(pr_rec)
            generated_bar = pr_rec

        else:
            generated_bar[generated_bar < 0.8] = 0
            generated_bar[generated_bar >= 0.8] = 64
            generated_bar = generated_bar.squeeze(0).detach().numpy()

        if i == 0:
            progression = generated_bar
        else :
            try:
                if not (progression[-16:,:] == generated_bar).all():
                    progression = np.concatenate((progression, generated_bar), axis=0)
            except:
                if all(all(x == generated_bar[i]) for i, x in enumerate(progression[-16:,:])):
                    progression = np.concatenate((progression, generated_bar), axis=0)

    # Use pypianoroll to export it in MIDI format
    track = pypianoroll.BinaryTrack(pianoroll=progression, program=0,
                              is_drum=False, name='Generated_interpolation')

    multtrack = pypianoroll.Multitrack(tracks=[track])
    #, tempo=np.asarray([90.0])) #, resolution=4, downbeat=[0, 16, 32, 48], name="Generated_interpolation")
    pypianoroll.write(
        path=f"output/generations/example_{arguments['--inputrep']}_{2}.mid", multitrack=multtrack)
