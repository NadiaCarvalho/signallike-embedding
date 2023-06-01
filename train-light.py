#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:12:43 2023

@author: carvalho

Usage:
    train.py [-h | --help]
    train.py [--version]
    train.py [--mps] [--gpu] [--gpudev GPUDEVICE] [--lr LR] [--maxiter MITER]
            [--runname RNAME] [--inputrep REP] [--path P] [--bsize BSIZE]
            [--nbframe NBFRAME] [--o OUT] [--save]

Options:
    -h --help  Show this helper
    --version  Show version and exit
    --mps  Use MPS or not [default: False]
    --gpu  Use GPU or not [default: False]
    --gpudev GPUDEVICE  Which GPU will be use [default: 0]
    --lr LR  Initial learning rate [default: 1e-4]
    --maxiter MITER  Maximum number of updates [default: 50]
    --runname RNAME  Set the name of the run for tensorboard [default: default_run]
    --inputrep REP  Set the representation which will be used as input [default: midilike]
    --path P  The path of the MIDI files folder (with a test and train folder) \
            [default: /fast-1/mathieu/datasets/Chorales_Bach_Proper_with_all_transposition].
    --bsize BSIZE  Batch size [default: 16]
    --nbframe NBFRAME  Number of frames per bar [default: 16]
    --o OUT  Path of the output directory [default: None]
    --save  Save the models during the training or not [default: True]
"""

import os
import time

import lightning as L
import models as m
import representations as rep_classes
import torch

from docopt import docopt
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


if __name__ == '__main__':

    arguments = docopt(__doc__, version='symbolic_embeddings v1.0')
    print(arguments)

    # Set detect anomaly
    torch.autograd.set_detect_anomaly(True)  # type: ignore

    # Parameters
    train_path = arguments['--path'] + '/train'
    test_path = arguments['--path'] + '/test'
    batch_size = int(arguments['--bsize'])
    nb_frame = int(arguments['--nbframe'])
    if arguments['--o'] == 'None':
        output_dr = os.getcwd() + '/output'
    else:
        output_dr = arguments['--o']

    # load the dataset
    if arguments['--inputrep'] == "pianoroll":
        dataset = rep_classes.Pianoroll(train_path, nbframe_per_bar=nb_frame)
        testset = rep_classes.Pianoroll(test_path, nbframe_per_bar=nb_frame)
        input_dim = 128
        seq_length = nb_frame
    elif arguments['--inputrep'] == "midilike":
        dataset = rep_classes.Midilike(train_path)
        testset = rep_classes.Midilike(test_path)
        input_dim = 1
    elif arguments['--inputrep'] == "midimono":
        dataset = rep_classes.Midimono(train_path)
        testset = rep_classes.Midimono(test_path)
        input_dim = 1
    elif arguments['--inputrep'] == "signallike":
        dataset = rep_classes.Signallike(
            train_path, nbframe_per_bar=nb_frame*2, mono=True)
        testset = rep_classes.Signallike(
            test_path, nbframe_per_bar=nb_frame*2, mono=True)
        input_dim = dataset.signal_size//64
    elif arguments['--inputrep'] == "notetuple":
        dataset = rep_classes.Notetuple(train_path)
        testset = rep_classes.Notetuple(test_path)
        input_dim = 5
    else:
        raise NotImplementedError(
            "Representation {} not implemented".format(arguments['--inputrep']))

    # Init the dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,  # type: ignore
                                              pin_memory=True, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4,  # type: ignore
                                              pin_memory=True, shuffle=False, drop_last=True)

    # Model parameters
    enc_hidden_size = 1024
    cond_hidden_size = 1024
    dec_hidden_size = 1024
    cond_outdim = 512
    num_layers_enc = 2
    num_layers_dec = 2
    num_subsequences = 4
    latent_size = 256

    if arguments['--inputrep'] in ['pianoroll', 'signallike']:
        output_dim = input_dim
    elif arguments['--inputrep'] == "midilike":
        output_dim = len(dataset.vocabulary)  # type: ignore
        seq_length = 64
    elif arguments['--inputrep'] == "midimono":
        output_dim = 130
        seq_length = 16
    elif arguments['--inputrep'] == "notetuple":
        output_dim = sum([len(v) for v in
                          dataset.vocabs]) + 129  # type: ignore
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
                               vocab=dataset.vocabs)  # type: ignore
    else:
        model = m.LightningVAE(encoder, decoder, arguments['--inputrep'])

    os.makedirs(f'{output_dr}/models/{arguments["--runname"]}/', exist_ok=True)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss',  # type: ignore
                                            save_top_k=1, mode='min',
                                            dirpath=f'{output_dr}/models/{arguments["--runname"]}/',
                                            filename=arguments['--runname'] + \
                                            '-{epoch}-{val_loss:.2f}',
                                            save_last=True),
        L.pytorch.callbacks.EarlyStopping(monitor='val_loss',  # type: ignore
                                          patience=5,
                                          mode='min'),
        L.pytorch.callbacks.LearningRateMonitor(  # type: ignore
            logging_interval='step'),
    ]

    trainer = L.Trainer(max_epochs=10, default_root_dir=output_dr,
                        enable_checkpointing=True, callbacks=callbacks)

    last_model = f'{output_dr}/models/{arguments["--runname"]}/last.ckpt'
    if os.path.exists(last_model):
        trainer.fit(model,
                    train_dataloaders=data_loader,
                    val_dataloaders=test_loader,
                    ckpt_path=last_model)
    else:
        trainer.fit(model,
                    train_dataloaders=data_loader,
                    val_dataloaders=test_loader)
