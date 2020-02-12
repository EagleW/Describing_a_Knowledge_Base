import torch.nn as nn
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from .baseRNN import BaseRNN
# self.word2idx['<UNK>'] = 1


class DecoderRNN(BaseRNN):

    def __init__(self, vocab_size, embedding, embed_size, pemsize, sos_id, eos_id,
                 unk_id, max_len=100, n_layers=1, rnn_cell='gru',
                 bidirectional=True, input_dropout_p=0, dropout_p=0,
                 lmbda=1.5, USE_CUDA = torch.cuda.is_available(), mask=0):
        hidden_size = embed_size

        super(DecoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.mask = mask
        self.embedding = embedding
        self.lmbda = lmbda
        self.USE_CUDA = USE_CUDA
        #directions
        self.Wh = nn.Linear(hidden_size * 2, hidden_size)
        #output
        self.V = nn.Linear(hidden_size * 3, self.output_size)
        #params for attention
        self.Wih = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder input
        self.Wfh = nn.Linear(hidden_size, hidden_size)  # for obtaining e from encoder field
        self.Ws = nn.Linear(hidden_size, hidden_size)  # for obtaining e from current state
        self.w_c = nn.Linear(1, hidden_size)  # for obtaining e from context vector
        self.v = nn.Linear(hidden_size, 1)
        # parameters for p_gen
        self.w_ih = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_fh = nn.Linear(hidden_size, 1)    # for changing context vector into a scalar
        self.w_s = nn.Linear(hidden_size, 1)    # for changing hidden state into a scalar
        self.w_x = nn.Linear(embed_size, 1)     # for changing input embedding into a scalar
        # parameters for self attention
        self_size = pemsize * 2  # hidden_size +
        self.wp = nn.Linear(self_size, self_size)
        self.wc = nn.Linear(self_size, self_size)
        self.wa = nn.Linear(self_size, self_size)

    def get_matrix(self, encoderp):
        tp = torch.tanh(self.wp(encoderp))
        tc = torch.tanh(self.wc(encoderp))
        f = tp.bmm(self.wa(tc).transpose(1, 2))
        return F.softmax(f, dim=2)

    def self_attn(self, f_matrix, encoderi, encoderf):
        c_contexti = torch.bmm(f_matrix, encoderi)
        c_contextf = torch.bmm(f_matrix, encoderf)
        return c_contexti, c_contextf

    def decode_step(self, input_ids, coverage, _h, enc_proj, batch_size, max_enc_len,
                    enc_mask, c_contexti, c_contextf, embed_input, max_source_oov, f_matrix):
        dec_proj = self.Ws(_h).unsqueeze(1).expand_as(enc_proj)
        cov_proj = self.w_c(coverage.view(-1, 1)).view(batch_size, max_enc_len, -1)
        e_t = self.v(torch.tanh(enc_proj + dec_proj + cov_proj).view(batch_size*max_enc_len, -1))

        # mask to -INF before applying softmax
        attn_scores = e_t.view(batch_size, max_enc_len)
        del e_t
        attn_scores.data.masked_fill_(enc_mask.data.byte(), -float('inf'))
        attn_scores = F.softmax(attn_scores, dim=1)

        contexti = attn_scores.unsqueeze(1).bmm(c_contexti).squeeze(1)
        contextf = attn_scores.unsqueeze(1).bmm(c_contextf).squeeze(1)

        # output proj calculation
        p_vocab = F.softmax(self.V(torch.cat((_h, contexti, contextf), 1)), dim=1)
        # p_gen calculation
        p_gen = torch.sigmoid(self.w_ih(contexti) + self.w_fh(contextf) + self.w_s(_h) + self.w_x(embed_input))
        p_gen = p_gen.view(-1, 1)
        weighted_Pvocab = p_vocab * p_gen
        weighted_attn = (1-p_gen) * attn_scores

        if max_source_oov > 0:
            # create OOV (but in-article) zero vectors
            ext_vocab = torch.zeros(batch_size, max_source_oov)
            if self.USE_CUDA:
                ext_vocab=ext_vocab.cuda()
            combined_vocab = torch.cat((weighted_Pvocab, ext_vocab), 1)
            del ext_vocab
        else:
            combined_vocab = weighted_Pvocab
        del weighted_Pvocab       # 'Recheck OOV indexes!'
        # scatter article word probs to combined vocab prob.
        combined_vocab = combined_vocab.scatter_add(1, input_ids, weighted_attn)
        return combined_vocab, attn_scores

    def forward(self, max_source_oov=0, targets=None, targets_id=None, input_ids=None,
                enc_mask=None, encoder_hidden=None, encoderi=None, encoderf=None,
                encoderp=None, teacher_forcing_ratio=None, w2fs=None, fig=False):

        targets, batch_size, max_length, max_enc_len = self._validate_args(targets, encoder_hidden, encoderi, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)
        coverage = torch.zeros(batch_size, max_enc_len)
        enci_proj = self.Wih(encoderi.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        encf_proj = self.Wfh(encoderf.view(batch_size*max_enc_len, -1)).view(batch_size, max_enc_len, -1)
        f_matrix = self.get_matrix(encoderp)
        enc_proj = enci_proj + encf_proj

        # get link attention scores
        c_contexti, c_contextf = self.self_attn(f_matrix, encoderi, encoderf)
        if self.USE_CUDA:
            coverage = coverage.cuda()
        if teacher_forcing_ratio:
            embedded = self.embedding(targets)
            embed_inputs = self.input_dropout(embedded)
            # coverage initially zero
            dec_lens = (targets > 0).float().sum(1)
            lm_loss, cov_loss = [], []
            hidden, _ = self.rnn(embed_inputs, decoder_hidden)
            # step through decoder hidden states
            for _step in range(max_length):
                _h = hidden[:, _step, :]
                target_id = targets_id[:, _step+1].unsqueeze(1)
                embed_input = embed_inputs[:, _step, :]

                combined_vocab, attn_scores = self.decode_step(input_ids, coverage, _h, enc_proj, batch_size,
                                                               max_enc_len, enc_mask, c_contexti, c_contextf,
                                                               embed_input, max_source_oov, f_matrix)
                # mask the output to account for PAD
                target_mask_0 = target_id.ne(0).detach()
                output = combined_vocab.gather(1, target_id).add_(sys.float_info.epsilon)
                lm_loss.append(output.log().mul(-1) * target_mask_0.float())

                coverage = coverage + attn_scores

                # Coverage Loss
                # take minimum across both attn_scores and coverage
                _cov_loss, _ = torch.stack((coverage, attn_scores), 2).min(2)
                cov_loss.append(_cov_loss.sum(1))
            # add individual losses
            total_masked_loss = torch.cat(lm_loss, 1).sum(1).div(dec_lens) + self.lmbda * \
                torch.stack(cov_loss, 1).sum(1).div(dec_lens)
            return total_masked_loss
        else:
            return self.evaluate(targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig)

    def evaluate(self, targets, batch_size, max_length, max_source_oov, c_contexti, c_contextf, f_matrix,
                 decoder_hidden, enc_mask, input_ids, coverage, enc_proj, max_enc_len, w2fs, fig):
        lengths = np.array([max_length] * batch_size)
        decoded_outputs = []
        if fig:
            attn = []
        embed_input = self.embedding(targets)
        # step through decoder hidden states
        for _step in range(max_length):
            _h, _c = self.rnn(embed_input, decoder_hidden)
            combined_vocab, attn_scores = self.decode_step(input_ids, coverage,
                                                           _h.squeeze(1), enc_proj, batch_size, max_enc_len, enc_mask,
                                                           c_contexti, c_contextf, embed_input.squeeze(1),
                                                           max_source_oov, f_matrix)
            # not allow decoder to output UNK
            combined_vocab[:, self.unk_id] = 0
            symbols = combined_vocab.topk(1)[1]
            if self.mask == 1:
                tmp_mask = torch.zeros_like(enc_mask, dtype=torch.uint8)
                for i in range(symbols.size(0)):
                    pos = (input_ids[i] == symbols[i]).nonzero()
                    if pos.size(0) != 0:
                        tmp_mask[i][pos] = 1
                enc_mask = torch.where(enc_mask > tmp_mask, enc_mask, tmp_mask)

            if fig:
                attn.append(attn_scores)
            decoded_outputs.append(symbols.clone())
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > _step) & eos_batches) != 0
                lengths[update_idx] = len(decoded_outputs)
            # change unk to corresponding field
            for i in range(symbols.size(0)):
                w2f = w2fs[i]
                if symbols[i].item() > self.vocab_size-1:
                    symbols[i] = w2f[symbols[i].item()]
            # symbols.masked_fill_((symbols > self.vocab_size-1), self.unk_id)
            embed_input = self.embedding(symbols)
            decoder_hidden = _c
            coverage = coverage + attn_scores
        if fig:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist(), f_matrix[0], \
                   torch.stack(attn, 1).squeeze(2)[0]
        else:
            return torch.stack(decoded_outputs, 1).squeeze(2), lengths.tolist()

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            h = self.Wh(h)
        return h

    def _validate_args(self, targets, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        else:
            max_enc_len = encoder_outputs.size(1)
        # inference batch size
        if targets is None and encoder_hidden is None:
            batch_size = 1
        else:
            if targets is not None:
                batch_size = targets.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default targets and max decoding length
        if targets is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")
            # torch.set_grad_enabled(False)
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if self.USE_CUDA:
                targets = targets.cuda()
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1     # minus the start of sequence symbol

        return targets, batch_size, max_length, max_enc_len
