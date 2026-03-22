from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer

dataset = load_dataset("bentrevett/multi30k")
# English to French translation dataset

class TranslationDataset(Dataset):
    def __init__(self,split,src_tokenizer,tgt_tokenizer):
        super().__init__()
        self.dataset = dataset[split]
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        if split == "train":
            src_sentences = [item['en'] for item in self.dataset]
            tgt_sentences = [item['de'] for item in self.dataset]
            src_tokenizer.build_vocab(src_sentences)
            tgt_tokenizer.build_vocab(tgt_sentences)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        
        item = self.dataset[idx]

        english_sentence = item['en']
        french_sentence = item['de']

        src_ids = self.src_tokenizer.encode(english_sentence)
        tgt_ids = self.tgt_tokenizer.encode(french_sentence)

        # add SOS and EOS to src too
        src_ids = [self.src_tokenizer.word2idx['<SOS>']] + src_ids + [self.src_tokenizer.word2idx['<EOS>']]

        decoder_input = [self.tgt_tokenizer.word2idx['<SOS>']] + tgt_ids
        decoder_target = tgt_ids + [self.tgt_tokenizer.word2idx['<EOS>']]

        return (
            torch.tensor(src_ids,dtype=torch.long), 
            torch.tensor(decoder_input,dtype=torch.long), 
            torch.tensor(decoder_target,dtype=torch.long)
        )
    

def collate_fn(batch):

    src_batch, decoder_input_batch, decoder_target_batch = zip(*batch)

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, 
        batch_first=True, 
        padding_value=0
    )

    decoder_input_batch = torch.nn.utils.rnn.pad_sequence(
        decoder_input_batch, 
        batch_first=True, 
        padding_value=0
    )

    decoder_target_batch = torch.nn.utils.rnn.pad_sequence(
        decoder_target_batch, 
        batch_first=True, 
        padding_value=0
    )

    return src_batch, decoder_input_batch, decoder_target_batch


