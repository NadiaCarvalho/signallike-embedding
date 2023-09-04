#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:41:24 2020

@author: prang
"""

import random
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import torch  # type: ignore
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn  # type: ignore


class Encoder_RNN(nn.Module):

    def __init__(self, input_dim, hidden_size, latent_size, num_layers,
                 dropout=0.5, packed_seq=False, device='cpu'):
        """ This initializes the encoder """
        super(Encoder_RNN, self).__init__()

        # Parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.packed_seq = packed_seq
        self.batch_first = True
        self.device = device

        # Layers
        self.RNN = nn.LSTM(input_dim, hidden_size, batch_first=self.batch_first,
                           num_layers=num_layers, bidirectional=True,
                           dropout=dropout)

    def forward(self, x, h0, c0, batch_size):

        # Pack sequence if needed
        if self.packed_seq:
            x = torch.nn.utils.rnn.pack_padded_sequence(x[0], x[1],
                                                        batch_first=self.batch_first,
                                                        enforce_sorted=False)
        # Forward pass
        _, (h, _) = self.RNN(x, (h0, c0))

        # Be sure to not have NaN values
        assert ((h == h).all()), 'NaN value in the output of the RNN, try to \
                                lower your learning rate'
        h = h.view(self.num_layers, 2, batch_size, -1)
        h = h[-1]
        h = torch.cat([h[0], h[1]], dim=1)

        return h

    def init_hidden(self, batch_size=1):
        # Bidirectional -> num_layers * 2
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size,
                            dtype=torch.float, device=self.device),) * 2


class Decoder_RNN_hierarchical(nn.Module):

    def __init__(self, input_size, latent_size, cond_hidden_size, cond_outdim,
                 dec_hidden_size, num_layers, num_subsequences, seq_length,
                 teacher_forcing_ratio=0, dropout=0.5):
        """ This initializes the decoder """
        super(Decoder_RNN_hierarchical, self).__init__()

        # Parameters
        self.num_subsequences = num_subsequences
        self.input_size = input_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.subseq_size = self.seq_length // self.num_subsequences

        # Layers
        self.tanh = nn.Tanh()
        self.fc_init_cond = nn.Linear(
            latent_size, cond_hidden_size * num_layers)
        self.conductor_RNN = nn.LSTM(latent_size // num_subsequences, cond_hidden_size,
                                     batch_first=True, num_layers=num_layers,
                                     bidirectional=False, dropout=dropout)
        self.conductor_output = nn.Linear(cond_hidden_size, cond_outdim)
        self.fc_init_dec = nn.Linear(cond_outdim, dec_hidden_size * num_layers)
        self.decoder_RNN = nn.LSTM(cond_outdim + input_size, dec_hidden_size,
                                   batch_first=True, num_layers=num_layers,
                                   bidirectional=False, dropout=dropout)
        self.decoder_output = nn.Linear(dec_hidden_size, input_size)

    def forward(self, latent, target, batch_size, teacher_forcing, device):

        # Get the initial state of the conductor
        h0_cond = self.tanh(self.fc_init_cond(latent))
        h0_cond = h0_cond.view(self.num_layers, batch_size, -1).contiguous()
        # Divide the latent code in subsequences
        latent = latent.view(batch_size, self.num_subsequences, -1)
        # Pass through the conductor
        subseq_embeddings, _ = self.conductor_RNN(latent, (h0_cond,)*2)
        subseq_embeddings = self.conductor_output(subseq_embeddings)

        # Get the initial states of the decoder
        h0s_dec = self.tanh(self.fc_init_dec(subseq_embeddings))
        h0s_dec = h0s_dec.view(self.num_layers, batch_size,
                               self.num_subsequences, -1).contiguous()
        # Init the output seq and the first token to 0 tensors
        out = torch.zeros(batch_size, self.seq_length, self.input_size,
                          dtype=torch.float, device=device)
        token = torch.zeros(batch_size, self.subseq_size, self.input_size,
                            dtype=torch.float, device=device)
        # Autoregressivly output tokens
        for sub in range(self.num_subsequences):
            subseq_embedding = subseq_embeddings[:, sub, :].unsqueeze(1)
            subseq_embedding = subseq_embedding.expand(
                -1, self.subseq_size, -1)
            h0_dec = h0s_dec[:, :, sub, :].contiguous()
            c0_dec = h0s_dec[:, :, sub, :].contiguous()
            # Concat the previous token and the current sub embedding as input
            dec_input = torch.cat((token, subseq_embedding), -1)
            # Pass through the decoder
            token, (h0_dec, c0_dec) = self.decoder_RNN(
                dec_input, (h0_dec, c0_dec))
            token = self.decoder_output(token)
            # Fill the out tensor with the token
            out[:, sub*self.subseq_size: ((sub+1)*self.subseq_size), :] = token
            # If teacher_forcing replace the output token by the real one sometimes
            if teacher_forcing:
                if random.random() <= self.teacher_forcing_ratio:
                    token = target[:, sub *
                                   self.subseq_size: ((sub+1)*self.subseq_size), :]
        return out


class VAE(nn.Module):

    def __init__(self, encoder, decoder, input_representation, teacher_forcing=True, device='cpu'):
        super(VAE, self).__init__()
        """ This initializes the complete VAE """

        # Parameters
        self.input_rep = input_representation
        self.tf = teacher_forcing
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.device = device

        # Layers
        self.hidden_to_mu = nn.Linear(
            2 * encoder.hidden_size, encoder.latent_size)
        self.hidden_to_sig = nn.Linear(
            2 * encoder.hidden_size, encoder.latent_size)

    def forward(self, x):

        if self.input_rep == 'notetuple':
            batch_size = x[0].size(0)
        else:
            batch_size = x.size(0)

        # Encoder pass
        h_enc, c_enc = self.encoder.init_hidden(batch_size)  # type: ignore
        hidden = self.encoder(x, h_enc, c_enc, batch_size)
        # Reparametrization
        mu = self.hidden_to_mu(hidden)
        sig = self.hidden_to_sig(hidden)
        eps = torch.randn_like(mu).detach().to(self.device)
        latent = (sig.exp().sqrt() * eps) + mu

        # Decoder pass
        if self.input_rep == 'midilike':
            # One hot encoding of the target for teacher forcing purpose
            target = torch.nn.functional.one_hot(x.squeeze(2).long(),
                                                 self.input_size).float()
            x_reconst = self.decoder(latent, target, batch_size,
                                     teacher_forcing=self.tf, device=self.device)
        else:
            x_reconst = self.decoder(latent, x, batch_size,
                                     teacher_forcing=self.tf, device=self.device)

        return mu, sig, latent, x_reconst

    def batch_pass(self, x, loss_fn, optimizer, w_kl, test=False):

        # Zero grad
        self.zero_grad()

        # Forward pass
        mu, sig, latent, x_reconst = self(x)

        # Compute losses
        kl_div = - 0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
        if self.input_rep in ["midilike", "MVAErep"]:
            reconst_loss = loss_fn(x_reconst.permute(
                0, 2, 1), x.squeeze(2).long())
        elif self.input_rep == "notetuple":
            x_reconst = x_reconst.permute(0, 2, 1)
            x_in, l = x
            loss_ts_maj = loss_fn(
                x_reconst[:, :len(self.vocab[0]), :],  # type: ignore
                x_in[:, :, 0].long())
            current = len(self.vocab[0])  # type: ignore

            loss_ts_min = loss_fn(
                x_reconst[:, current:current +
                          len(self.vocab[1]), :],  # type: ignore
                x_in[:, :, 1].long())
            current += len(self.vocab[1])  # type: ignore

            loss_pitch = loss_fn(
                x_reconst[:, current:current + 129, :], x_in[:, :, 2].long())
            current += 129

            loss_dur_maj = loss_fn(
                x_reconst[:, current:current +
                          len(self.vocab[2]), :],  # type: ignore
                x_in[:, :, 3].long())
            current += len(self.vocab[2])  # type: ignore

            loss_dur_min = loss_fn(
                x_reconst[:, current:current +
                          len(self.vocab[3]), :],  # type: ignore
                x_in[:, :, 4].long())
            reconst_loss = loss_ts_maj + loss_ts_min + \
                loss_pitch + loss_dur_maj + loss_dur_min
        else:
            reconst_loss = loss_fn(x_reconst, x)

        # Backprop and optimize
        if not test:
            loss = reconst_loss + (w_kl * kl_div)
            loss.backward()
            optimizer.step()
        else:
            loss = reconst_loss + kl_div

        return loss, kl_div, reconst_loss

    def generate(self, latent):

        # Create dumb target
        input_shape = (1, self.decoder.seq_length, self.decoder.input_size)
        db_trg = torch.zeros(input_shape)  # type: ignore
        # Forward pass in the decoder
        generated_bar = self.decoder(latent.unsqueeze(0), db_trg, batch_size=1,
                                     device=self.device, teacher_forcing=False)

        return generated_bar


class LightningVAE(L.LightningModule):

    def __init__(self, encoder, decoder, input_representation, vocab=None, teacher_forcing=True):
        super(LightningVAE, self).__init__()
        """ This initializes the complete VAE """
        # Parameters
        self.input_rep = input_representation
        self.tf = teacher_forcing
        self.encoder = encoder
        self.decoder = decoder

        self.w_kl = 0

        self.vocab = vocab
        if input_representation == 'notetuple' and vocab is None:
            raise ValueError(
                'Vocab must be provided for notetuple input representation')

        # Layers
        self.hidden_to_mu = torch.nn.Linear(
            2 * encoder.hidden_size, encoder.latent_size)
        self.hidden_to_sig = torch.nn.Linear(
            2 * encoder.hidden_size, encoder.latent_size)

        if input_representation in ['pianoroll', 'signallike', 'dft128']:
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def forward(self, x):

        if self.input_rep == 'notetuple':
            batch_size = x[0].size(0)
        else:
            batch_size = x.size(0)

        # Encoder pass
        h_enc, c_enc = self.encoder.init_hidden(batch_size)
        hidden = self.encoder(x, h_enc, c_enc, batch_size)

        # Reparametrization
        mu = self.hidden_to_mu(hidden)
        sig = self.hidden_to_sig(hidden)
        eps = torch.randn_like(mu).detach().to(self.device)
        latent = (sig.exp().sqrt() * eps) + mu

        # Decoder pass
        if self.input_rep == 'midilike':
            # One hot encoding of the target for teacher forcing purpose
            target = torch.nn.functional.one_hot(x.squeeze(2).long(),
                                                 self.input_size).float()
            x_reconst = self.decoder(latent, target, batch_size,
                                     teacher_forcing=self.tf, device=self.device)
        else:
            x_reconst = self.decoder(latent, x, batch_size,
                                     teacher_forcing=self.tf, device=self.device)

        return mu, sig, latent, x_reconst

    def notetuple_reconstruction_loss(self, x_reconst, x):
        """Compute the reconstruction loss for a
        given input in notetuple format and its reconstruction"""
        x_reconst = x_reconst.permute(0, 2, 1)
        x_in, l = x
        loss_ts_maj = self.loss_fn(
            x_reconst[:, :len(self.vocab[0]), :],  # type: ignore
            x_in[:, :, 0].long())
        current = len(self.vocab[0])  # type: ignore

        loss_ts_min = self.loss_fn(
            x_reconst[:, current:current +
                      len(self.vocab[1]), :],  # type: ignore
            x_in[:, :, 1].long())
        current += len(self.vocab[1])  # type: ignore

        loss_pitch = self.loss_fn(
            x_reconst[:, current:current + 129, :],
            x_in[:, :, 2].long())
        current += 129

        loss_dur_maj = self.loss_fn(
            x_reconst[:, current:current +
                      len(self.vocab[2]), :],  # type: ignore
            x_in[:, :, 3].long())
        current += len(self.vocab[2])  # type: ignore

        loss_dur_min = self.loss_fn(
            x_reconst[:, current:current +
                      len(self.vocab[3]), :],  # type: ignore
            x_in[:, :, 4].long())
        reconst_loss = loss_ts_maj + loss_ts_min + \
            loss_pitch + loss_dur_maj + loss_dur_min

        return reconst_loss

    def compute_reconstruction_loss(self, x, x_reconst):
        """ Compute the reconstruction loss for a given input and its reconstruction """

        if self.input_rep in ["midilike", "MVAErep"]:
            reconst_loss = self.loss_fn(x_reconst.permute(
                0, 2, 1), x.squeeze(2).long())
        elif self.input_rep == "notetuple":
            reconst_loss = self.notetuple_reconstruction_loss(x_reconst, x)
        else:  # pianoroll, signallike, dft128
            reconst_loss = self.loss_fn(x_reconst, x)

        return reconst_loss

    def compute_reconstructions_accuracy(self, reconst_loss, threshold=0.5):
        """ Compute the reconstruction accuracy for a given input and its reconstruction """
        accuracy = (reconst_loss < threshold).float().mean().item() # Calculate accuracy
        return accuracy * 100.0  # Convert to percentage

    def training_step(self, batch, batch_idx):

        x = batch

        # Zero grad
        self.zero_grad()

        # Forward pass
        mu, sig, _, x_reconst = self(x)

        # Compute losses
        kl_div = - 0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
        reconst_loss = self.compute_reconstruction_loss(x, x_reconst)

        # Backprop and optimize
        loss = reconst_loss + (self.w_kl * kl_div)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_kl_div", kl_div, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_reconst_loss", reconst_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "kl_div": kl_div, "reconst_loss": reconst_loss}

    def validation_step(self, batch, batch_idx):
        x = batch

        # Zero grad
        self.zero_grad()

        # Forward pass
        mu, sig, _, x_reconst = self(x)

        # Compute losses
        kl_div = - 0.5 * torch.sum(1 + sig - mu.pow(2) - sig.exp())
        reconst_loss = self.compute_reconstruction_loss(x, x_reconst)

        # Backprop and optimize
        loss = reconst_loss + kl_div
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_kl_div", kl_div, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_reconst_loss", reconst_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "kl_div": kl_div, "reconst_loss": reconst_loss}

    def generate(self, latent):
        # Create dumb target
        input_shape = (1, self.decoder.seq_length, self.decoder.input_size)
        db_trg = torch.zeros(input_shape)
        # Forward pass in the decoder
        return self.decoder(latent.unsqueeze(0), db_trg, batch_size=1,
                            device=self.device, teacher_forcing=False)

    def interpolate_from_points(self, starting_point, ending_point, nb_points=10):
        # Get the corresponging latent code
        _, _, st_latent, _ = self(starting_point.unsqueeze(0))
        _, _, end_latent, _ = self(ending_point.unsqueeze(0))
        st_latent = st_latent.squeeze(0).detach().numpy()
        end_latent = end_latent.squeeze(0).detach().numpy()
        # Interpolate between this two coordinates
        return torch.tensor(np.linspace(st_latent, end_latent, nb_points))

    def on_test_batch_end(self, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called when the test batch ends."""
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def increase_w_kl(self, epoch) -> None:
        """
        Increase the weight of the KL divergence loss
        """
        if self.input_rep in ["pianoroll"]:
            if epoch < 150 and epoch > 0 and epoch % 10 == 0:
                self.w_kl += 1e-5
            elif epoch > 150 and epoch % 10 == 0:
                self.w_kl += 1e-4
        elif self.input_rep in ["midilike", "signallike", "dft128"] and epoch % 10 == 0 and epoch > 0:
            self.w_kl += 1e-8
        elif self.input_rep == "midimono" and epoch % 10 == 0 and epoch > 0:
            self.w_kl += 1e-4
        elif self.input_rep == "notetuple" and epoch % 10 == 0 and epoch > 0:
            self.w_kl += 1e-6

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """On load checkpoint callback"""
        super().on_load_checkpoint(checkpoint)

        for i in range(checkpoint['epoch']):
            self.increase_w_kl(i)

        print('Starting with w_kl', self.w_kl)

    def on_train_epoch_end(self) -> None:
        """
        Called when the epoch ends.
        """
        self.increase_w_kl(self.current_epoch)
        self.log("w_kl", self.w_kl, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
