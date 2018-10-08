import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from predictor import Predictor
from utils.loader import Table2text_seq
from structure_generator.EncoderRNN import EncoderRNN
from structure_generator.DecoderRNN import DecoderRNN
from structure_generator.seq2seq import Seq2seq
from eval_final import Evaluate
from eval import Evaluate_test


class Config(object):
    cell = "GRU"
    emsize = 256
    pemsize = 5
    nlayers = 1
    lr = 0.001
    epochs = 19
    batch_size = 300
    dropout = 0
    bidirectional = False
    max_grad_norm = 10
    max_len = 100


class ConfigTest(object):
    cell = "GRU"
    emsize = 30
    pemsize = 30
    nlayers = 1
    lr = 0.001
    epochs = 2
    batch_size = 10
    dropout = 0
    bidirectional = True
    max_grad_norm = 1
    testmode = True
    max_len = 50


parser = argparse.ArgumentParser(description='pointer model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='params.pkl',
                    help='path to save the final model')
parser.add_argument('--mode', type=int,  default=0,
                    help='train(0)/predict_individual(1)/predict_file(2)/compute score(3) or keep train (4)')
parser.add_argument('--type', type=int,  default=0,
                    help='person(0)/animal(1)')
parser.add_argument('--mask', type=int,  default=0,
                    help='false(0)/true(1)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
config = Config()
# config = ConfigTest()

if args.mask == 1:
    filepost = "_m"
else:
    filepost = ""

if args.type == 1:
    args.save = 'params_D.pkl'
    config.epochs = 20
    filepost += "_A.txt"
else:
    filepost += "_P.txt"
t_dataset = Table2text_seq(0, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
v_dataset = Table2text_seq(1, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
print("number of training examples: %d" % t_dataset.len)
embedding = nn.Embedding(t_dataset.vocab.size, config.emsize, padding_idx=0)
encoder = EncoderRNN(t_dataset.vocab.size, embedding, config.emsize, t_dataset.max_p, config.pemsize,
                     input_dropout_p=config.dropout, dropout_p=config.dropout, n_layers=config.nlayers,
                     bidirectional=config.bidirectional, rnn_cell=config.cell, variable_lengths=True)
decoder = DecoderRNN(t_dataset.vocab.size, embedding, config.emsize, config.pemsize, sos_id=3, eos_id=2, unk_id=1,
                     n_layers=config.nlayers, rnn_cell=config.cell, bidirectional=config.bidirectional,
                     input_dropout_p=config.dropout, dropout_p=config.dropout, USE_CUDA=args.cuda, mask=args.mask)
model = Seq2seq(encoder, decoder).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
predictor = Predictor(model, v_dataset.vocab, args.cuda)


def train_batch(dataset, batch_idx, model, teacher_forcing_ratio):
    batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov = \
        dataset.get_batch(batch_idx)
    losses = model(batch_s, batch_o_s, max_source_oov, batch_f, batch_pf, batch_pb, source_len, batch_t,
                   batch_o_t, teacher_forcing_ratio)
    batch_loss = losses.mean()
    model.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return batch_loss.item(), len(source_len)


def train_epoches(t_dataset, v_dataset, model, n_epochs, teacher_forcing_ratio):
    eval_f = Evaluate_test()
    best_dev = 0
    train_loader = t_dataset.corpus
    len_batch = len(train_loader)
    epoch_examples_total = t_dataset.len
    for epoch in range(1, n_epochs + 1):
        model.train(True)
        torch.set_grad_enabled(True)
        epoch_loss = 0
        for batch_idx in range(len_batch):
            loss, num_examples = train_batch(t_dataset, batch_idx, model, teacher_forcing_ratio)
            epoch_loss += loss * num_examples
            sys.stdout.write(
                '%d batches processed. current batch loss: %f\r' %
                (batch_idx, loss)
            )
            sys.stdout.flush()
        epoch_loss /= epoch_examples_total
        log_msg = "Finished epoch %d with losses: %.4f" % (epoch, epoch_loss)
        print(log_msg)
        predictor = Predictor(model, v_dataset.vocab, args.cuda)
        print("Start Evaluating")
        cand, ref = predictor.preeval_batch(v_dataset)
        print('Result:')
        print('ref: ', ref[1][0])
        print('cand: ', cand[1])
        final_scores = eval_f.evaluate(live=True, cand=cand, ref=ref)
        epoch_score = 2*final_scores['ROUGE_L']*final_scores['Bleu_4']/(final_scores['Bleu_4']+ final_scores['ROUGE_L'])
        if epoch_score > best_dev:
            torch.save(model.state_dict(), args.save)
            print("model saved")
            best_dev = epoch_score


if __name__ == "__main__":
    if args.mode == 0:
        # train
        try:
            print("start training...")
            train_epoches(t_dataset, v_dataset, model, config.epochs, 1)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    elif args.mode == 1:
        # predict sentence
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        dataset = Table2text_seq(2, type=args.type, USE_CUDA=args.cuda, batch_size=1)
        print("Read test data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        while True:
            seq_str = input("Type index from (%d to %d) to continue:\n" %(0, dataset.len - 1))
            i = int(seq_str)
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, sources, targets, fields, list_oovs, source_len\
                , max_source_oov, w2fs = dataset.get_batch(i)
            table = []
            for i in range(len(sources[0])):
                table.append(fields[0][i])
                table.append(":")
                table.append(sources[0][i])
            print("Table:")
            print(' '.join(table)+'\n')
            print("Refer: ")
            print(' '.join(targets[0])+'\n')
            outputs = predictor.predict(batch_s, batch_o_s, batch_f, batch_pf, batch_pb, max_source_oov
                                        , source_len, list_oovs[0], w2fs)
            print("Result: ")
            print(outputs)
            print('-'*120)
    elif args.mode == 2:
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        dataset = Table2text_seq(2, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
        print("Read test data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        print("number of test examples: %d" % dataset.len)
        print("Start Evaluating")
        lines = predictor.predict_file(dataset)
        print("Start writing")
        f_out = open("Output" + filepost, 'w')
        f_out.writelines(lines)
        f_out.close()
    elif args.mode == 3:
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        dataset = Table2text_seq(2, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
        print("Read test data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        print("number of test examples: %d" % dataset.len)
        eval_f = Evaluate()
        print("Start Evaluating")
        cand, ref = predictor.preeval_batch(dataset)
        scores = []
        fields = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"]
        final_scores = eval_f.evaluate(live=True, cand=cand, ref=ref)

        f_out = open("score" + filepost, 'w')
        for field in fields:
            f_out.write(field + '\t' + str(final_scores[field])+'\n')
        f_out.close()
    elif args.mode == 4:
        # load and keep training
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        # train
        try:
            print("start training...")
            train_epoches(t_dataset, v_dataset, model, 1, 1)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        dataset = Table2text_seq(2, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
        print("Read test data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        print("number of test examples: %d" % dataset.len)
        eval_f = Evaluate()
        print("Start Evaluating")
        cand, ref = predictor.preeval_batch(dataset)
        scores = []
        fields = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"]
        final_scores = eval_f.evaluate(live=True, cand=cand, ref=ref)
        x = input('Save (1) or not')
        if x == '1':
            torch.save(model.state_dict(), args.save)
            print("model saved")

