import torch
from collections import Counter
import pickle


class Vocabulary:
    """Vocabulary class for mapping between words and ids"""
    def __init__(self, word2idx=None, idx2word=None, field=None, corpus=None, max_words=50000, min_frequency=5,
                 start_end_tokens=True):
        if corpus is None:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.start_end_tokens = False
            self.size = len(word2idx)
        else:
            self.word2idx = dict()
            self.idx2word = dict()
            self.size = 0
            self.vocabulary = None
            # most common words
            self.max_words = max_words
            # least common words
            self.min_frequency = min_frequency
            self.start_end_tokens = start_end_tokens
            self._build_vocabulary(corpus, field)
            print("Finish build vocabulary")
            self._build_word_index()
            print("Finish build word dictionary")

    def _build_vocabulary(self, corpus, field):
        vocabulary = Counter(word for sent in corpus for word in sent)
        field_vocab = Counter(word for sent in field for word in sent)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        self.vocabulary = Counter(vocabulary)
        self.vocabulary.update(field_vocab)
        self.size = len(self.vocabulary) + 2  # padding and unk tokens
        if self.start_end_tokens:
            self.size += 2

    def _build_word_index(self):
        self.word2idx['<UNK>'] = 1
        self.word2idx['<PAD>'] = 0

        if self.start_end_tokens:
            self.word2idx['<EOS>'] = 2
            self.word2idx['<SOS>'] = 3

        offset = len(self.word2idx)
        for idx, word in enumerate(self.vocabulary):
            self.word2idx[word] = idx + offset
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add_start_end(self, vector):
        vector.append(self.word2idx['<EOS>'])
        return [self.word2idx['<SOS>']] + vector

    def vectorize_field(self, vector):
        _o_field = []
        for word in vector:
            _o_field.append(self.word2idx[word])
        return _o_field

    def vectorize_source(self, vector, table):
        source_oov = []
        _o_source, _source = [], []
        for word in vector:
            try:
                _o_source.append(self.word2idx[word])
                _source.append(self.word2idx[word])
            except KeyError:
                if word not in source_oov:
                    _o_source.append(len(source_oov) + self.size)
                    source_oov.append(word)
                else:
                    _o_source.append(source_oov.index(word) + self.size)
                _source.append(self.word2idx[table[word]])
        return _o_source, source_oov, _source

    def vectorize_target(self, vector, source_oov, table):
        _o_target, _target = [], []
        for word in vector:
            try:
                _o_target.append(self.word2idx[word])
                _target.append(self.word2idx[word])
            except KeyError:
                if word not in source_oov:
                    # print(word)
                    _o_target.append(self.word2idx['<UNK>'])
                    _target.append(self.word2idx['<UNK>'])
                else:
                    _o_target.append(source_oov.index(word) + self.size)
                    _target.append(self.word2idx[table[word]])
        return self.add_start_end(_o_target), self.add_start_end(_target)


class Table2text_seq:
    def __init__(self, mode, type=0, batch_size=128, USE_CUDA=torch.cuda.is_available()):
        self.type = type
        self.vocab = None
        # self.target_vocab = None
        self.text_len = 0
        self.max_p = 0
        self.mode = mode
        self.batch_size = batch_size
        self.USE_CUDA = USE_CUDA
        if mode == 0:
            if self.type == 0:
                path = "train_P.pkl"
            else:
                path = "train_A.pkl"
        elif mode == 1:
            if self.type == 0:
                path = "valid_P.pkl"
            else:
                path = "valid_A.pkl"
        else:
            if self.type == 0:
                path = "test_P.pkl"
            else:
                path = "test_A.pkl"
        self.data = self.load_data(path)
        self.len = len(self.data)
        self.corpus = self.batchfy()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

    def load_data(self, path):
        # (qkey, qitem, index)
        with open(path, 'rb') as output:
            data = pickle.load(output)
        old_sources = data["source"]
        old_targets = data["target"]
        total = []
        samples = []
        total_field = []
        for idx, old_source in enumerate(old_sources):
            source = []
            field = []
            table = {}
            p_for = []
            p_bck = []
            target = old_targets[idx]
            if len(target) > self.text_len:
                self.text_len = len(target) + 2
            for key, value, index in old_source:
                # change key into special tokens
                tag = '<'+key+'>'
                source.append(value)
                field.append(tag)
                p_for.append(index)
                if value not in table:
                    table[value] = tag
            curr_p_max = max(p_for) + 1
            for p in p_for:
                p_bck.append(curr_p_max - p)
            if self.max_p < curr_p_max:
                self.max_p = curr_p_max
            total.append(source + target)
            total_field.append(field)
            samples.append([source, target, field, p_for, p_bck, table])
        samples.sort(key=lambda x: len(x[0]), reverse=True)
        if self.type == 0:
            vocab_path = "vocab.pkl"
        else:
            vocab_path = "vocab_D.pkl"
        if self.mode == 0:
            if self.type == 0:
                self.vocab = Vocabulary(corpus=total, field=total_field)
            else:
                self.vocab = Vocabulary(corpus=total, field=total_field, min_frequency=0)
            data = {
                "idx2word": self.vocab.idx2word,
                "word2idx": self.vocab.word2idx
            }
            with open(vocab_path, 'wb') as output:
                pickle.dump(data, output)
        else:
            with open(vocab_path, 'rb') as output:
                data = pickle.load(output)
            self.vocab = Vocabulary(word2idx=data["word2idx"], idx2word=data["idx2word"])
        return samples

    def batchfy(self):
        samples = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        corpus = []
        for sample in samples:
            corpus.append(self.vectorize(sample))
        return corpus

    def pad_vector(self, vector, maxlen):
        padding = maxlen - len(vector)
        vector.extend([0] * padding)
        return vector

    def vectorize(self, sample):
        # batch_s--> tensor batch of table with ids
        # batch_o_s --> tensor batch of table with ids and <unk> replaced by temp OOV ids
        # batch_t--> tensor batch of text with ids
        # batch_o_t --> tensor batch of target and <unk> replaced by temp OOV ids
        # batch_f--> tensor batch of field with ids(might not exist)
        # batch_pf--> tensor batch of forward position
        # batch_pb--> tensor batch of backward position
        # batch_o_f --> tensor batch of field and used wordid
        # max_article_oov --> max number of OOV tokens in article batch

        batch_o_s, batch_o_t, batch_f, batch_t, batch_s, batch_pf, batch_pb = [], [], [], [], [], [], []
        source_len, target_len, w2fs = [], [], []
        list_oovs = []
        targets = []
        sources = []
        fields = []
        max_source_oov = 0
        for data in sample:
            source = data[0]
            target = data[1]
            field = data[2]
            p_for = data[3]
            p_bck = data[4]
            table = data[5]
            source_len.append(len(source))
            target_len.append(len(target) + 2)
            _o_source, source_oov, _source = self.vocab.vectorize_source(source, table)
            # print(source_oov)
            _o_target, _target = self.vocab.vectorize_target(target, source_oov, table)
            _o_fields = self.vocab.vectorize_field(field)
            if max_source_oov < len(source_oov):
                max_source_oov = len(source_oov)
            if self.mode != 0:
                oovidx2word = {idx: word for idx, word in enumerate(source_oov)}
                w2f={(idx+self.vocab.size): self.vocab.word2idx[table[word]] for idx, word in enumerate(source_oov)}
                w2fs.append(w2f)
                list_oovs.append(oovidx2word)
                targets.append(target)
                sources.append(source)
                fields.append(field)
            batch_o_s.append(_o_source)
            batch_pf.append(p_for)
            batch_pb.append(p_bck)
            batch_o_t.append(_o_target)
            batch_t.append(_target)
            batch_s.append(_source)
            batch_f.append(_o_fields)

        batch_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_s]
        batch_o_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_o_s]
        batch_f = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_f]
        batch_pf = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pf]
        batch_pb = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pb]

        batch_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_t]
        batch_o_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_o_t]

        batch_o_s = torch.stack(batch_o_s, dim=0)
        batch_f = torch.stack(batch_f, dim=0)
        batch_pf = torch.stack(batch_pf, dim=0)
        batch_pb = torch.stack(batch_pb, dim=0)
        batch_t = torch.stack(batch_t, dim=0)
        batch_s = torch.stack(batch_s, dim=0)
        batch_o_t = torch.stack(batch_o_t, dim=0)
        if self.mode != 0:
            targets= [i[:max(target_len)-2] for i in targets]
            sources= [i[:max(source_len)] for i in sources]
            fields = [i[:max(source_len)] for i in fields]
            return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, targets, sources, fields, list_oovs, source_len, \
                max_source_oov, w2fs
        else:
            return batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov

    def get_batch(self, index):
        if self.mode == 0:
            batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov \
                = self.corpus[index]
            batch_s = batch_s.to(self.device)
            batch_o_s = batch_o_s.to(self.device)
            batch_t = batch_t.to(self.device)
            batch_o_t = batch_o_t.to(self.device)
            batch_f = batch_f.to(self.device)
            batch_pf = batch_pf.to(self.device)
            batch_pb = batch_pb.to(self.device)
            return batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov
        else:
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, targets, sources, fields, list_oovs, source_len, \
                max_source_oov, w2fs = self.corpus[index]
            batch_s = batch_s.to(self.device)
            batch_o_s = batch_o_s.to(self.device)
            batch_f = batch_f.to(self.device)
            batch_pf = batch_pf.to(self.device)
            batch_pb = batch_pb.to(self.device)
            return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, sources, targets, fields, list_oovs, source_len, \
                max_source_oov, w2fs

