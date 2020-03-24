import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import librosa
from librosa.feature import melspectrogram

class TIMIT(Dataset):
    _ext_txt = ".TXT"
    _ext_phone = ".PHN"
    _ext_audio = ".WAV"
    _UNK_LABEL = '_UNK_'

    def __init__(self, root, text_vocab=None, phone_vocab=None):
        self._path = root
        walker = []
        for curr_root, _, fnames in sorted(os.walk(self._path)):
            for fname in fnames:
                if fname.endswith(self._ext_audio):
                    walker.append(os.path.join(curr_root, fname[:-len(self._ext_audio)]))

        self._walker = list(walker)
        if text_vocab is None:
            text_vocab = []
            for fileid in walker:
                audio_text = self.load_txt_item(fileid)
                for t in audio_text:
                    if t not in text_vocab:
                        text_vocab.append(t)

        if phone_vocab is None:
            phone_vocab = []
            for fileid in walker:
                audio_phone = self.load_phone_item(fileid)
                for p in audio_phone:
                    if p not in phone_vocab:
                        phone_vocab.append(p)

        self._text_vocab = text_vocab
        self._phone_vocab = phone_vocab

        self._text_vocab2id = {t:i for i, t in enumerate(text_vocab)}
        self._phone_vocab2id = {p:i for i, p in enumerate(phone_vocab)}

    def __getitem__(self, n):
        fileid = self._walker[n]
        return self.load_timit_item(fileid)

    def __len__(self):
        return len(self._walker)

    def load_txt_item(self, fileid):
        file_text = fileid + self._ext_txt
        with open(file_text) as ft:
            audio_text = ft.readline().strip().lower().split()[:2]
        return audio_text

    def load_phone_item(self, fileid):
        file_phone = fileid + self._ext_phone
        with open(file_phone) as fp:
            audio_phone = [line.strip().split()[-1] for line in fp]
        return audio_phone

    def load_timit_item(self, fileid):
        file_audio = fileid + self._ext_audio

        waveform, sample_rate = librosa.load(file_audio)
        mel_specgram = melspectrogram(y=waveform, sr=sample_rate)
        mel_specgram = mel_specgram.swapaxes(0, 1)

        audio_text = [self._text_vocab2id[t] if t in self._text_vocab2id else self._UNK_LABEL
                      for t in self.load_txt_item(fileid)]
        audio_phone = [self._phone_vocab2id[p] if p in self._phone_vocab2id else self._UNK_LABEL
                      for p in self.load_phone_item(fileid)]

        return {
            'mel_specgram':torch.from_numpy(mel_specgram),
            'audio_text':torch.tensor(audio_text).long(),
            'audio_phone':torch.tensor(audio_phone).long()
        }

def variable_collate_fn(batch):
    pad_token_id = -1
    mel_specgram = []
    audio_text = []
    audio_phone = []

    for sample in batch:
        mel_specgram.append(sample['mel_specgram'])
        audio_text.append(sample['audio_text'])
        audio_phone.append(sample['audio_phone'])

    mel_specgram = pad_sequence(mel_specgram, batch_first=True, padding_value=pad_token_id)
    audio_text = pad_sequence(audio_text, batch_first=True, padding_value=pad_token_id)
    audio_phone = pad_sequence(audio_phone, batch_first=True, padding_value=pad_token_id)

    mel_specgram_mask = mel_specgram.ne(pad_token_id)

    return {
        'mel_specgram': mel_specgram,
        'audio_text': audio_text,
        'audio_phone': audio_phone,
        'mel_specgram_mask': mel_specgram_mask
    }
if __name__ == '__main__':
    timit_dataset = TIMIT('/Users/atulkumar/Downloads/TIMIT/TRAIN/')
    ''' 
    for i in range(len(timit_dataset)):
        sample = timit_dataset[i]
        print(i, sample['mel_specgram'].size())
        if i == 3:
            break
    '''

    dataloader = DataLoader(timit_dataset, batch_size=4, shuffle=True,
                            num_workers=1, collate_fn=variable_collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, len(sample_batched['mel_specgram']),  sample_batched['utterance_id'])
        if i_batch == 3:
            break
