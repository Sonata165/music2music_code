import os
import sys
from transformers import AutoTokenizer

def main():
    # train_tokenizer()
    # test_tokenizer()
    # train_tokenizer2()
    # train_tokenizer3()
    # add_more_bar_tokens()
    # add_chord_tokens()
    # adjust_pad_token()
    add_texture_tokens()

def adjust_pad_token():
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")

    tk.pad_token = '[PAD]'

    tk.push_to_hub("LongshenOu/m2m_ft")

def observe_tokens():
    data_fp = '/data1/longshen/musecoco_data/datasets/train.txt'
    with open(data_fp) as f:
        data = f.readlines()

    data = [i.strip().split() for i in data]
    tokens_of_two_bar = []
    cnt = 0 
    for i in data:
        # print(i)
        cnt += len(i)
        tokens_of_two_bar.append(len(i))
    print(cnt)

    # print quantiles of tokens of two_bar
    from numpy import percentile
    print('quantiles of tokens of two_bar')
    print('min:', min(tokens_of_two_bar))
    print('max:', max(tokens_of_two_bar))
    print('median:', percentile(tokens_of_two_bar, 50))
    print('75th percentile:', percentile(tokens_of_two_bar, 75))
    print('90th percentile:', percentile(tokens_of_two_bar, 90))
    print('95th percentile:', percentile(tokens_of_two_bar, 95))
    print('99th percentile:', percentile(tokens_of_two_bar, 99))

def test_tokenizer():
    # from tokenizers import Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")
    inp = "o-0 i-128 p-172 d-1 b-1 o-6 i-33 p-19 d-2 b-2 i-128 p-170 d-1 b2 p-164 d-1 o-12 i-33 p-19 d-2 i-128 p-170 d-1 p-168 d-1 o-17 i-88 p-79 d-17 o-18 i-0 p-79 d-3 i-2 p-67 d-2 i-33 p-19 d-8 i-128 p-170 d-1 o-24 i-128 p-170 d-1 p-164 d-1 o-30 i-29 p-60 d-5 i-33 p-19 d-6 i-52 p-60 d-5 p-55 d-5 p-52 d-6 i-128 p-174 d-1 p-164 d-1 o-36 i-29 p-62 d-6 i-33 p-21 d-12 i-52 p-62 d-5 p-57 d-5 p-53 d-6 i-128 p-168 d-1 o-39 i-128 p-168 d-1 o-42 i-29 p-64 d-9 i-52 p-64 d-8 p-60 d-8 p-55 d-6 i-128 p-171 d-1 b-1 o-0 i-33 p-24 d-13 i-44 p-64 d-57 p-60 d-57 i-128 p-177 d-1 p-164 d-1 o-6 i-26 p-76 d-4 p-72 d-4 p-67 d-4 i-128 p-170 d-1 o-10 i-26 p-76 d-2 p-72 d-3 p-67 d-3 o-12 i-2 p-60 d-6 p-55 d-17 p-52 d-4 i-26 p-76 d-2 p-72 d-1 p-67 d-1 p-60 d-5 p-55 d-15 i-128 p-170 d-1 p-168 d-1 o-17 i-26 p-59 d-13 p-50 d-13 o-18 i-2 p-59 d-14 p-50 d-14 i-26 p-74 d-2 p-71 d-1 p-67 d-1 i-33 p-24 d-17 i-128 p-170 d-1 p-164 d-1 o-23 i-26 p-67 d-1 p-62 d-1 o-24 i-26 p-69 d-6 i-128 p-170 d-1 o-25 i-26 p-64 d-6 o-30 i-26 p-67 d-1 p-62 d-2 i-29 p-60 d-5 i-52 p-60 d-6 p-55 d-5 p-52 d-6 i-128 p-174 d-1 p-164 d-1 o-35 i-26 p-67 d-1 p-62 d-1 o-36 i-26 p-69 d-13 p-64 d-13 i-29 p-62 d-5 i-52 p-62 d-5 p-57 d-5 p-53 d-5 i-128 p-170 d-1 p-168 d-1 o-42 i-29 p-64 d-12 i-33 p-24 d-5 i-52 p-64 d-12 p-60 d-12 p-55 d-11 i-128 p-170 d-1 o-47 i-2 p-57 d-9 p-53 d-9 p-52 d-9 p-48 d-9 b-1"
    out = tokenizer(inp)['input_ids']
    print(out)
    print(tokenizer.decode(out))
    print(len(out))
    # out = tokenizer.e ([inp])
    # print(out)


def add_texture_tokens():
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_ft")

    # Chord roots
    txt_tokens = ['txt-{}'.format(i) for i in range(0, 10)]

    tk.add_tokens(txt_tokens)

    tk.push_to_hub("LongshenOu/m2m_ft_tokenizer_txt")


def add_chord_tokens():
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")

    # Chord roots
    chord_roots = [
        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    chord_types = [
        'Major', 'Minor', 'Augmented', 'Diminished', 'Major7', 'Minor7', 'Dominant7', 'Sus4', 'Sus2'
    ]

    tk.add_tokens(chord_roots)
    tk.add_tokens(chord_types)

    tk.push_to_hub("LongshenOu/m2m_ft")


def add_more_bar_tokens():
    tk = AutoTokenizer.from_pretrained("LongshenOu/m2m_pt")
    tk.add_tokens(['b-{}'.format(i) for i in range(1, 33)])
    tk.push_to_hub("LongshenOu/m2m_ft")


def train_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer

    paths = ['/data1/longshen/musecoco_data/datasets/train.txt']

    from tokenizers.trainers import BpeTrainer
    trainer = BpeTrainer(special_tokens=["[EOS]", "[BOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    from tokenizers.pre_tokenizers import WhitespaceSplit
    tokenizer.pre_tokenizer = WhitespaceSplit()

    tokenizer.train(paths, trainer)
    
    # os.mkdir('test_tokenizer')
    tokenizer.save("./test_tokenizer/tokenizer-wiki.json")

def train_tokenizer2():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    paths = ['/data1/longshen/musecoco_data/datasets/train.txt']

    special_tokens = ["[EOS]", "[BOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.BpeTrainer(special_tokens=special_tokens)

    # Define a custom pre-tokenizer class
    class CustomPreTokenizer(pre_tokenizers.PreTokenizer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def pre_tokenize_str(self, text):
            return text.split()

    # Create an instance of the custom pre-tokenizer
    custom_pre_tokenizer = CustomPreTokenizer()

    tokenizer.pre_tokenizer = custom_pre_tokenizer

    tokenizer.train(paths, trainer)

    # Save the tokenizer
    tokenizer.save("./test_tokenizer/tokenizer-wiki.json")

def train_tokenizer3():
    from tokenizers import Tokenizer
    from pathlib import Path
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel, BPE
    from tokenizers.pre_tokenizers import WhitespaceSplit

    # We need to specify the UNK token
    new_tokenizer = Tokenizer(model=WordLevel(
        unk_token="[UNK]",
    ))
    new_tokenizer.add_special_tokens([
        '[EOS]', '[BOS]', '[UNK]', "[CLS]", "[SEP]", "[PAD]", "[MASK]",
        '[HIST]', '[PITCH]', '[INST]', '[MELODY]', '[CHORD]'
        ])
    new_tokenizer.add_tokens(['b-1'] +
                             [f'o-{i}' for i in range(128)] +
                             [f'i-{i}' for i in range(129)] +
                             [f'p-{i}' for i in range(256)] +
                             [f'd-{i}' for i in range(128)] +
                             [f'v-{i}' for i in range(32)] +
                             [f's-{i}' for i in range(254)] +
                             [f't-{i}' for i in range(49)]
    )
                             
    # Add pretokenizer
    paths = ['/data1/longshen/musecoco_data/datasets/train.txt']

    # trainer = BpeTrainer(special_tokens=["[EOS]", "[BOS]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    new_tokenizer.pre_tokenizer = WhitespaceSplit()

    # tokenizer.train(paths, trainer)
    
    # os.mkdir('test_tokenizer')
    new_tokenizer.save("./test_tokenizer/musecoco.json")

def convert_to_pretrained_tokenizer():
    '''
    Look for demo pages' tokenizer behavior, then
    customize the PretrainedTokenizerBase.build_inputs_with_special_tokens
    if needed

    or use this
    https://discuss.huggingface.co/t/gpt2tokenizer-not-putting-bos-eos-token/27394/2

    same eos and bos ...?
    '''
    pass




if __name__ == '__main__':
    main()