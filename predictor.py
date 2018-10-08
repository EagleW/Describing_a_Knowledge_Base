import torch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  #TkAgg
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

class Predictor(object):
    def __init__(self, model, vocab, USE_CUDA):
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.USE_CUDA = USE_CUDA

    def predict(self, batch_s, batch_o_s, batch_f, batch_pf, batch_pb, max_source_oov, source_len, list_oovs, w2fs):
        torch.set_grad_enabled(False)
        decoded_outputs, lengths = self.model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb,
                                              source_len, w2fs=w2fs)
        length = lengths[0]
        output = []
        # print(decoded_outputs)
        for i in range(length):
            symbol = decoded_outputs[0][i].item()
            if symbol < self.vocab.size:
                output.append(self.vocab.idx2word[symbol])
            else:
                output.append(list_oovs[symbol-self.vocab.size])
        print(len(output))
        return ' '.join([i for i in output if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])

    def overlap(self, line):
        indexrefer = []
        indexoutput = []
        for word in line['sources']:
            index = [i for i, j in enumerate(line['refer']) if j.lower() == word.lower()]
            indexrefer.append(index)
            index = [i for i, j in enumerate(line['output']) if j.lower() == word.lower()]
            indexoutput.append(index)
        line['indexrefer'] = indexrefer
        line['indexoutput'] = indexoutput
        new_table = []
        for field in line['fields']:
            new_table.append(field[field.find('<')+1:field.find('>')])
        line['fields'] = new_table

    def predict_file(self, dataset):
        torch.set_grad_enabled(False)
        i = 0
        lines = []
        for batch_idx in range(len(dataset.corpus)):
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, sources, targets, fields, list_oovs, source_len, \
                max_source_oov, w2fs = dataset.get_batch(batch_idx)
            decoded_outputs, lengths = self.model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb,
                                                  source_len, w2fs=w2fs)
            pos = batch_pf.tolist()
            for j in range(len(lengths)):
                line = {}
                line['sources'] = sources[j]
                line['fields'] = fields[j]
                i += 1
                line['refer'] = targets[j]
                line['pos'] = [p for p in pos[j] if p != 0]
                out_seq = []
                for k in range(lengths[j]):
                    symbol = decoded_outputs[j][k].item()
                    if symbol < self.vocab.size:
                        out_seq.append(self.vocab.idx2word[symbol])
                    else:
                        out_seq.append(list_oovs[j][symbol-self.vocab.size])
                line['output'] = out_seq
                self.overlap(line)

                if i % 2500 == 0:
                    print("Percentages:  %.4f" % (i/float(dataset.len)))
                lines.append(str(line)+'\n')
        return lines

    def preeval_batch(self, dataset):
        torch.set_grad_enabled(False)
        refs = {}
        cands = {}
        i = 0
        for batch_idx in range(len(dataset.corpus)):
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, _, targets, _, list_oovs, source_len, max_source_oov, w2fs = dataset.get_batch(batch_idx)
            decoded_outputs, lengths = self.model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb, source_len, w2fs=w2fs)
            for j in range(len(lengths)):
                i += 1
                ref = self.prepare_for_bleu(targets[j])
                refs[i] = [ref]
                out_seq = []
                for k in range(lengths[j]):
                    symbol = decoded_outputs[j][k].item()
                    if symbol < self.vocab.size:
                        out_seq.append(self.vocab.idx2word[symbol])
                    else:
                        out_seq.append(list_oovs[j][symbol-self.vocab.size])
                out = self.prepare_for_bleu(out_seq)
                cands[i] = out

                # if i % 2500 == 0:
                #     print("Percentages:  %.4f" % (i/float(dataset.len)))
        return cands, refs

    def prepare_for_bleu(self, sentence):
        sent=[x for x in sentence if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
        sent = ' '.join(sent)
        return sent

    def showAttention(self, input_words, output_words, attentions, name, type):
        # Set up figure with colorbar
        # pp = PdfPages(name)
        plt.rcParams.update({'font.size': 18})
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap=plt.cm.Blues)
        if type == 1:
            fig.colorbar(cax, shrink=0.7, pad=0.03)
        else:
            fig.colorbar(cax, pad=0.03)

        # Set up axes
        ax.set_xticklabels([''] + input_words, rotation=90)
        ax.set_yticklabels([''] + output_words)
        if type == 0:
            ax.set_xlabel("Table Position")
            ax.set_ylabel("Table Position")
        else:
            ax.set_xlabel("Structured KB")
            ax.set_ylabel("Output")

        # ax.set_xticklabels([''] + input.split(' ') +
        #                    ['<EOS>'], rotation=90)
        # ax.set_yticklabels(output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # plt.show()
        fig.savefig(name, bbox_inches='tight')
        plt.close(fig)

    def figure(self, batch_s, batch_o_s, batch_f, batch_pf, batch_pb, max_source_oov, source_len, list_oovs, w2fs, type,
               visual):
        torch.set_grad_enabled(False)
        decoded_outputs, lengths, self_matrix, soft = self.model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb,
                                              source_len, w2fs=w2fs, fig=True)
        length = lengths[0]
        output = []
        # print(decoded_outputs)
        for i in range(length):
            symbol = decoded_outputs[0][i].item()
            if symbol < self.vocab.size:
                output.append(self.vocab.idx2word[symbol])
            else:
                output.append(list_oovs[symbol-self.vocab.size])
        output = [i for i in output if i != '<PAD>' and i != '<EOS>' and i != '<SOS>']
        print(self_matrix.size(), soft.size())
        pos = [str(i) for i in batch_pf[0].cpu().tolist()]
        combine = []
        for j in range(len(pos)):
            combine.append(visual[j] + " : " + pos[j])
        self.showAttention(pos, combine, self_matrix.cpu(), 'self.png', 0)
        self.showAttention(type, output[19:25], soft[19:25].cpu(), 'type.png', 1)
        # return output