import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

    # Tokenize source and target
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    # Truncate if needed to fit within seq_len (accounting for special tokens)
        max_enc_len = self.seq_len - 2  # [SOS] + [EOS]
        max_dec_len = self.seq_len - 1  # [SOS] only (label gets [EOS] later)

        enc_input_tokens = enc_input_tokens[:max_enc_len]
        dec_input_tokens = dec_input_tokens[:max_dec_len]

    # Add special tokens
        encoder_input = [self.tokenizer_src.token_to_id('[SOS]')] + enc_input_tokens + [self.tokenizer_src.token_to_id('[EOS]')]
        decoder_input = [self.tokenizer_tgt.token_to_id('[SOS]')] + dec_input_tokens
        label = dec_input_tokens + [self.tokenizer_tgt.token_to_id('[EOS]')]

    # Pad if needed
        pad_id_src = self.tokenizer_src.token_to_id('[PAD]')
        pad_id_tgt = self.tokenizer_tgt.token_to_id('[PAD]')

        encoder_input += [pad_id_src] * (self.seq_len - len(encoder_input))
        decoder_input += [pad_id_tgt] * (self.seq_len - len(decoder_input))
        label += [pad_id_tgt] * (self.seq_len - len(label))

        return {
        "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
        "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
        "encoder_mask": (torch.tensor(encoder_input, dtype=torch.long) != pad_id_src).unsqueeze(0).unsqueeze(0).int(),
        "decoder_mask": (torch.tensor(decoder_input, dtype=torch.long) != pad_id_tgt).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len),
        "label": torch.tensor(label, dtype=torch.long),
        "src_text": src_text,
        "tgt_text": tgt_text
        }


    
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0