import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import json

class TIMIT(Dataset):
    _ext_txt = ".TXT"
    _ext_phone = ".PHN"
    _ext_word = ".WRD"
    _ext_audio = ".WAV"
    _BLANK_LABEL = '_B_'
    '''
    mapping from:
    Lee, K. and Hon, H. Speaker-independent phone recognition using hidden markov models. 
    IEEE Transactions on Acoustics, Speech, and Signal Processing, 1989.
    '''
    _phone_map_48 ={
        'ux':'uw',
        'axr':'er',
        'em':'m',
        'nx':'n',
        'eng':'ng',
        'hv':'hh',
        'pcl': 'cl',
        'tcl': 'cl',
        'kcl': 'cl',
        'qcl': 'cl',
        'bcl': 'vcl',
        'gcl': 'vcl',
        'dcl': 'vcl',
        'h#':'sil',
        '#h':'sil',
        'pau':'sil',
        'ax-h':'ax',
        "q": None
    }
    _phone_map_39 = {
        'cl':'sil',
        'vcl': 'sil',
        'epi': 'sil',
        'el':'l',
        'en':'n',
        'sh':'zh',
        'ao':'aa',
        'ih':'ix',
        'ah':'ax'
    }

    def __init__(self, root):
        self._path = root
        walker = []
        for curr_root, _, fnames in sorted(os.walk(root)):
            for fname in fnames:
                if fname.endswith(self._ext_phone):
                    walker.append(os.path.join(curr_root, fname[:-len(self._ext_phone)]))

        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        return self.load_timit_item(fileid)

    def __len__(self):
        return len(self._walker)

    def init_dataset(self, root_dir):
        self.read_stats(root_dir)
        self.read_vocab(root_dir)

    def get_audio_features(self, fileid):
        '''
        Standard speech preprocessing was applied to transform the audio files into feature sequences. 26 channel
        mel-frequency filter bank and a pre-emphasis coefficient of 0.97 were used to compute 12 mel-frequency
        cepstral coefficients plus an energy coefficient on 25ms Hamming windows at 10ms intervals.
        Delta coefficients were added to create input sequences of length 26 vectors, and all coefficient
        were normalised to have mean zero and standard deviation one over the training set.
        '''
        file_audio = fileid + self._ext_audio
        d, sr = librosa.load(file_audio, sr=None)
        d = librosa.effects.preemphasis(d)
        hop_length = int(0.010 * sr)
        n_fft = int(0.025 * sr)
        mfcc = librosa.feature.mfcc(d, sr, n_mfcc=13,
                                     hop_length=hop_length,
                                     n_fft=n_fft, window='hamming')
        mfcc[0] = librosa.feature.rms(y=d,
                                       hop_length=hop_length,
                                       frame_length=n_fft)
        deltas1 = librosa.feature.delta(mfcc, order=1)
        deltas2 = librosa.feature.delta(mfcc, order=2)
        mfccs_plus_deltas = np.vstack([mfcc, deltas1, deltas2])

        #individual wave file normalization - power norm might does this so not needed
        #mfccs_plus_deltas -= (np.mean(mfccs_plus_deltas, axis=0) + 1e-8)
        mfccs_plus_deltas = mfccs_plus_deltas.swapaxes(0, 1)
        return mfccs_plus_deltas

    def dump_phone_vocab(self, root_dir):
        walker = []
        for p in ['TRAIN']:
            for curr_root, _, fnames in sorted(os.walk(os.path.join(root_dir, p))):
                for fname in fnames:
                    if fname.endswith(self._ext_phone):
                        walker.append(os.path.join(curr_root, fname[:-len(self._ext_phone)]))

        phone_vocab = []
        for fileid in walker:
            audio_phones = self.load_phone_item(fileid)
            for audio_phone in audio_phones:
                if audio_phone['phone'] not in phone_vocab:
                    phone_vocab.append(audio_phone['phone'])
        with open(os.path.join(root_dir, 'phone.vocab'), 'w', encoding='utf8') as f:
            f.write('\n'.join(sorted(phone_vocab)))

    def dump_mean_var(self, root_dir):
        walker = []
        for p in ['TRAIN']:
            for curr_root, _, fnames in sorted(os.walk(os.path.join(root_dir, p))):
                for fname in fnames:
                    if fname.endswith(self._ext_phone):
                        walker.append(os.path.join(curr_root, fname[:-len(self._ext_phone)]))

        all_mfccs = []
        for fileid in walker:
            mfcc = self.get_audio_features(fileid)
            mfcc = np.vsplit(mfcc, mfcc.shape[0])
            all_mfccs.extend(mfcc)

        all_mfccs = np.stack(all_mfccs, axis=0).squeeze(axis=1)
        mean = np.expand_dims(all_mfccs.mean(axis=0), axis=0)
        var = np.expand_dims(all_mfccs.var(axis=0), axis=0)
        np.savez(os.path.join(root_dir, 'stats.npz'), mean=mean, var=var)

    def read_stats(self, root_dir):
        train_stats = np.load(os.path.join(root_dir, 'stats.npz'))
        self._mean = np.expand_dims(train_stats['mean'], axis=0)
        self._var = np.expand_dims(train_stats['var'], axis=0)

    def normalize(self, mfcc):
        return (mfcc - self._mean) / self._var

    def read_vocab(self, root_dir):
        phone_vocab = []
        for w in open(os.path.join(root_dir, 'phone.vocab'), 'r', encoding='utf8'):
            if w:
                phone_vocab.append(w)

        self._phone_vocab = phone_vocab
        self._phone_vocab2id = {p:i for i, p in enumerate(phone_vocab)}

    def load_txt_item(self, fileid):
        file_text = fileid + self._ext_txt
        with open(file_text) as ft:
            audio_text = ft.readline().strip().lower().split()
        return {
            'start': int(audio_text[0]),
            'end'  : int(audio_text[1]),
            'words': audio_text[2:]
        }

    def load_word_item(self, fileid):
        file_text = fileid + self._ext_word
        audio_words = []
        with open(file_text) as ft:
            for line in ft:
                parts = line.strip().lower().split()
                audio_word = {
                    'start': int(parts[0]),
                    'end'  : int(parts[1]),
                    'word' : parts[2]
                }
                audio_words.append(audio_word)

        return audio_words

    def load_phone_item(self, fileid):
        file_phone = fileid + self._ext_phone
        audio_phones = []
        with open(file_phone) as fp:
            for line in fp:
                parts = line.strip().split()
                phone = parts[2]
                if phone in self._phone_map_48:
                    phone = self._phone_map_48[phone]
                if phone is None:
                    continue
                if phone in self._phone_map_39:
                    phone = self._phone_map_39[phone]
                audio_phone = {
                    'start': int(parts[0]),
                    'end': int(parts[1]),
                    'phone': phone
                }
                audio_phones.append(audio_phone)
        return audio_phones

    def load_timit_item(self, fileid):
        mfccs = self.get_audio_features(fileid)
        length = len(mfccs)
        mfccs = self.normalize(mfccs)
        mfccs = np.expand_dims(mfccs, axis=0)
        audio_phones = self.load_phone_item(fileid)
        phones = [self._phone_vocab2id[audio_phone['phone']] for audio_phone in audio_phones]

        return {
            'mfcc':torch.from_numpy(mfccs),
            'length': torch.LongTensor([length]),
            'phone':torch.LongTensor(phones)
        }

def variable_collate_fn(batch):
    pad_token_id = -1
    mfccs = []
    phones = []
    lengths = []
    for sample in batch:
        mfccs.append(sample['mfcc'])
        lengths.append(sample['length'])
        phones.append(sample['phone'].unsqueeze(0))

    mfccs = pad_sequence(mfccs, batch_first=True, padding_value=pad_token_id)
    lengths = torch.cat(lengths)
    phones = pad_sequence(phones, batch_first=True, padding_value=pad_token_id)

    return {
        'mfcc': mfccs,
        'length': lengths,
        'phone': phones
    }
if __name__ == '__main__':

    timit_dataset = TIMIT(os.path.join(os.path.expanduser('~'),
                                       'neural_sequence_transduction/TIMIT/TRAIN/'))
    '''
    timit_dataset.dump_phone_vocab()
    timit_dataset.dump_mean_var(os.path.join(os.path.expanduser('~'),
                                             'neural_sequence_transduction/TIMIT/'))
    timit_dataset.init_dataset(os.path.join(os.path.expanduser('~'),
                                             'neural_sequence_transduction/TIMIT/'))
     '''
    dataloader = DataLoader(timit_dataset, batch_size=4, shuffle=True,
                            num_workers=1, collate_fn=variable_collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['mel_specgram'].size(),  sample_batched['phones'].size())
        break

