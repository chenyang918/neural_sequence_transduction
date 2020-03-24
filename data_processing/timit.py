import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import librosa
from librosa.feature import melspectrogram

class TIMIT(Dataset):
    _ext_txt = ".TXT"
    _ext_phone = ".PHN"
    _ext_word = ".WRD"
    _ext_audio = ".WAV"
    _UNK_LABEL = '_UNK_'

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

    def dump_phone_vocab(self, root_dir):
        walker = []
        for p in ['TRAIN', 'TEST']:
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
        file_audio = fileid + self._ext_audio

        waveform, sample_rate = librosa.load(file_audio)
        mel_specgram = melspectrogram(y=waveform, sr=sample_rate)
        mel_specgram = mel_specgram.swapaxes(0, 1)
        text_item = self.load_txt_item(fileid)
        mel_specgram = mel_specgram[text_item['start']:text_item['end']+1, :]
        ''' 
        audio_text = [self._text_vocab2id[t] if t in self._text_vocab2id else self._UNK_LABEL
                      for t in self.load_txt_item(fileid)]
        audio_phone = [self._phone_vocab2id[p] if p in self._phone_vocab2id else self._UNK_LABEL
                      for p in self.load_phone_item(fileid)]
        '''
        return {
            'mel_specgram':torch.from_numpy(mel_specgram)
        }

def variable_collate_fn(batch):
    pad_token_id = -1
    mel_specgram = []


    for sample in batch:
        mel_specgram.append(sample['mel_specgram'])

    mel_specgram = pad_sequence(mel_specgram, batch_first=True, padding_value=pad_token_id)
    mel_specgram_mask = mel_specgram.ne(pad_token_id)

    return {
        'mel_specgram': mel_specgram,
        'mel_specgram_mask': mel_specgram_mask
    }
if __name__ == '__main__':
    timit_dataset = TIMIT(os.path.join(os.path.expanduser('~'),
                                       'neural_sequence_transduction/TIMIT/TRAIN/'))
    timit_dataset.dump_phone_vocab(os.path.join(os.path.expanduser('~'),
                                       'neural_sequence_transduction/TIMIT/'))
    ''' 
    dataloader = DataLoader(timit_dataset, batch_size=4, shuffle=True,
                            num_workers=1, collate_fn=variable_collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['mel_specgram'].size())
        if i_batch == 3:
            break
    '''