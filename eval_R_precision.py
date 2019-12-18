from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from nltk import RegexpTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import random
import argparse
import sys
import os

from encoder import RNN_ENCODER, CNN_ENCODER
from config import cfg, cfg_from_file



if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

WORDS_NUM = 18
USE_CUDA = True

def is_contain_chinese(token):
    for ch in token:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RP. Use build_RPData.py to prepare RP_DATA directory.')
    parser.add_argument('-fake_dir', dest='data_dir', type=str, help='Specify path to RP_DATA directory.')
    parser.add_argument('-cap_path', dest='caption_pkl', type=str, help='Specify path to captions.pickle, from AttnGan')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args



# used to feed the R-Precision Test
class EvalRPDataset(Dataset):
    def __init__(self, fake_dir, caption_pkl_path):
        assert os.path.isdir(fake_dir)
        assert os.path.isfile(caption_pkl_path)

        self.length = len([name for name in os.listdir(fake_dir)
                           if (os.path.isfile(os.path.join(fake_dir, name))) and (name.endswith('png'))])
        print('Total of {} sample found.'.format(self.length))

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.data_dir = fake_dir
        self.caption_pkl = caption_pkl_path

        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(fake_dir)

    def load_text_data(self, data_dir):
        filepath = self.caption_pkl
        if not os.path.isfile(filepath):
            print("Needs original captions.pickle file for indexing.")
            print(filepath)
            quit()
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)

        all_captions = self.load_caption(data_dir, wordtoix)
        all_captions_new = []
        for file in all_captions:
            new_file_texts = []
            for sent in file:
                if len(sent) == 0:
                    print('length of this sent is 0: {}'.format(sent))
                    raise Exception
                rev = []
                for w in sent:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                    else:
                        if w is 'taupecolored':
                            rev.append(wordtoix['taupe'])
                        elif w is 'yellowgreen':
                            rev.append(wordtoix['yellow'])
                        elif w is 'allblack':
                            rev.append(wordtoix['black'])
                        elif w is 'spottedwhite':
                            rev.append(wordtoix['white'])
                        print('Did not found "{}" in wordtoix dict'.format(w))
                new_file_texts.append(rev)
            all_captions_new.append(new_file_texts)

        return all_captions_new, ixtoword, wordtoix, n_words


    def load_caption(self, data_dir, wordtoix):
        def tokenize(in_captions):
            res = []
            for cap in in_captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())

                if len(tokens) == 0:
                    print('cap:', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0 and not (is_contain_chinese(t) or not t in wordtoix):
                        tokens_new.append(t)
                    else:
                        print('ERROR!!!')
                        correct_tokens = []
                        correct_str = ''
                        for item in tokens:
                            item = item.replace('锟斤拷', ' ')
                            correct_str += item
                        correct_tokens.append(correct_str)
                        break

                if is_contain_chinese(t) or not t in wordtoix:
                    for t in correct_tokens[0].split(' '):
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)


                res.append(tokens_new)
            return res

        all_captions = []
        for i in range(self.length):
            cap_path = os.path.join(data_dir, "{}.txt".format(i))
            with open(cap_path, 'r') as f:
                captions = f.read().split('\n') #[:100]
                # tokenize
                captions = tokenize(captions)
                all_captions.append(captions)

        return all_captions


    def get_caption(self, fileno):
        all_caps = self.captions[fileno]
        ret_cap = []
        ret_caplen = []

        for cap in all_caps:
            sent_caption = np.asarray(cap).astype('int64')
            if (sent_caption == 0).sum() > 0:
                print('ERROR: do not need END (0) token', sent_caption)
            num_words = len(sent_caption)
            # pad with 0s
            x = np.zeros((WORDS_NUM, 1), dtype='int64')
            if num_words < WORDS_NUM:
                x[:num_words, 0] = sent_caption
                x_len = num_words
            else:
                ix = list(np.arange(num_words))
                np.random.shuffle(ix)
                ix = ix[:WORDS_NUM]
                ix = np.sort(ix)
                x[:, 0] = sent_caption[ix]
                x_len = WORDS_NUM
            ret_cap.append(x)
            ret_caplen.append(x_len)

        return ret_cap, ret_caplen


    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, '{}.png'.format(index))
        image = self.norm(Image.open(image_path).convert('RGB'))
        image = image.unsqueeze(0)
        captions, cap_lens = self.get_caption(index)
        return image, captions, cap_lens


    def __len__(self):
        return self.length


def prepare_data(data):
    imgs, captions, captions_lens = data

    # convert list to tensor
    captions = torch.from_numpy(np.array(captions))
    captions_lens = torch.from_numpy(np.array(captions_lens))

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    # get where the real sentence got sorted to
    real_index = (sorted_cap_indices == 0).nonzero()

    real_imgs = []
    for i in range(len(imgs)):
        # imgs[i] = imgs[i][sorted_cap_indices]
        if USE_CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    # sent_indices = sent_indices[sorted_cap_indices]
    if USE_CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens, real_index]


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def evaluateSimilarity(dataset, cnn_model, rnn_model, R=10):
    # data: image, captions, cap_lens
    cnn_model.eval()
    rnn_model.eval()

    c_success = 0
    c_total = 0
    print('length of dataset is {}'.format(len(dataset)))
    for i in tqdm(range(len(dataset))):
        img, captions, cap_lens, real_index = prepare_data(dataset[i])

        _, sent_code = cnn_model(img[-1].unsqueeze(0))
        hidden = rnn_model.init_hidden(100)
        _, sent_emb = rnn_model(captions, cap_lens, hidden)

        sim = cosine_similarity(sent_code, sent_emb)
        real_sim = sim[real_index]
        sim, _ = sim.sort(descending=True)
        useful_sim = sim[:R]
        success = (real_sim in useful_sim)
        # success = (sim.max() == sim[real_index])
        if bool(success):
            c_success += 1
        c_total += 1

    rp = c_success / c_total
    return rp


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(5450, nhidden=256)        # dataset.n_words
    image_encoder = CNN_ENCODER(256)     # cfg.TEXT.EMBEDDING_DIM
    labels = Variable(torch.LongTensor(range(BATCH_SIZE)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)

    if USE_CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch

if __name__ == '__main__':
    args = parse_args()
    BATCH_SIZE = 64
    args.fake_dir = 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\attnGAN_Results\\bird_AttnGAN2\\RP_data'
    args.cap_path = 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\Datasets\\attnGAN_data\\birds\\captions.pickle'
    args.manualSeed = 1900

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    if USE_CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    torch.cuda.set_device(args.gpu_id)
    cudnn.benchmark = True

    # Get dataloader
    assert os.path.isfile(args.cap_path), "specify captions.pickle file"
    fake_dataset = EvalRPDataset(args.fake_dir, args.cap_path)

    # res = fake_dataset.get_caption(73)

    assert fake_dataset

    # Train
    text_encoder, image_encoder, labels, start_epoch = build_models()
    # print(fake_dataset.n_words)     # 5450 words

    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    this_model_name = args.fake_dir.split('\\')[-2]
    print("RP: {}%".format(evaluateSimilarity(fake_dataset, image_encoder, text_encoder) * 100))

##############################################################################################
#
#   R Precision:      AttnGAN 32_64:    16.37%(R=1) 41.95%(R=5)  48.37%(R=7)    55.67%(R=10)
#
#
##############################################################################################
