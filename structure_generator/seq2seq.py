import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb, input_lengths=None,
                target=None, target_id=None, teacher_forcing_ratio=0, w2fs=None, fig=False):
        encoderi, encoderf, encoderp, encoder_hidden, mask = self.encoder(batch_s, batch_f, batch_pf, batch_pb,
                                                                          input_lengths)
        result = self.decoder(max_source_oov=max_source_oov,
                              targets=target,
                              targets_id=target_id,
                              input_ids=batch_o_s,
                              enc_mask=mask,
                              encoder_hidden=encoder_hidden,
                              encoderi=encoderi,
                              encoderf=encoderf,
                              encoderp=encoderp,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              w2fs=w2fs,
                              fig=fig)
        return result
