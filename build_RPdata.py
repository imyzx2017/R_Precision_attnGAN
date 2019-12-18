import argparse
import os
import random
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import shutil
from nltk import RegexpTokenizer
import pickle

NUM_DIR = 50

def read_all_txt_orig(directory):
    all_s = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if not file.endswith(".txt"):
            continue
        with open(full_path) as f:
            captions = f.read().split('\n')
            for cap in captions:
                if len(cap) == 0 or len(cap) == 1:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    if t == 'thisbirdhasadarkgreybelly':
                        print(123)
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_s.append(" ".join(tokens_new) + "\n")
    return all_s


def read_all_from_pickle(filepath):
    all_s = []
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
    train_ix_list, test_ix_list = x[0], x[1]
    ixtoword = x[2]
    wordtoix = x[3]
    print('yellow' in wordtoix)

    for i in range(len(train_ix_list)):
        this_sent = []
        for t in train_ix_list[i]:
            this_sent.append(ixtoword[t])
        all_s.append(" ".join(this_sent) + "\n")

    for i in range(len(test_ix_list)):
        this_sent = []
        for t in test_ix_list[i]:
            this_sent.append(ixtoword[t])
        all_s.append(" ".join(this_sent) + "\n")


    return all_s


def generateSample(dir, rand=False):
    count = 0
    res = []

    files = os.listdir(dir)
    if rand:
        random.shuffle(files)

    for each in files:
        fname, fexp = os.path.splitext(each)
        fake = Image.open(os.path.join(dir, each)).convert('RGB')

        text = ''
        if args.text != '':
            # TODO: CHANGE THIS TO FIT YOUR NAMING
            txt_file = os.path.join(args.text, os.path.basename(dir), "{}.txt".format(fname.split("-")[0][:-2]))
            assert os.path.isfile(txt_file), txt_file
            # 选择一个和fake图像匹配的文本描述
            sentence_num = int(fname[-1])  # TODO: CHANGE THIS TO FIT YOUR NAMING
            with open(txt_file) as f:
                all_texts = f.readlines()
                text = all_texts[sentence_num]
                if 'thisshort' in text:
                    print(123)

        res.append((fake, text, all_texts))
        count += 1

    return res


def sampleFakeSentence(all_text, exclude_text):
    exclude_text = [x.strip() for x in exclude_text]
    res = []
    while len(res) < 99:
        tmp = random.choice(all_text)
        if tmp not in exclude_text:
            res.append(tmp)
            exclude_text.append(tmp)
    assert len(res) == 99
    return np.array(res)


def saveTestFiles(data, out, all_caps):
    out_path = os.path.join(out)
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    count = 0
    for d in tqdm(data):
        img, real, all_txts = d
        if 'thisshort' in real:
            print(123)
        # sample 99 fake sentences
        # fakes = np.random.choice(all_caps, 99, replace=False)
        fakes = sampleFakeSentence(all_caps, all_txts)

        fakes = [l + "\n" for l in fakes]
        texts = [real] + fakes
        # save files
        img.save(os.path.join(out_path, "{}.png".format(count)))
        with open(os.path.join(out_path, "{}.txt".format(count)), mode="w") as f:
            f.writelines(texts)
        count += 1
    print("RP Test Folder created at " + out_path)


def main(dirs, out, rand=False, all_caps_file=""):
    global NUM_DIR
    NUM_DIR = min(len(dirs), NUM_DIR)
    if rand:
        random.shuffle(dirs)
    # select first <NUM_DIR> DIR
    dirs = dirs[:NUM_DIR]
    data = []
    for directory in tqdm(dirs):
        data += generateSample(directory)

    print("Making R Precision Test Directory.")
    with open(all_caps_file) as f:
        all_caps = f.readlines()
    all_caps = [l.strip() for l in all_caps]
    saveTestFiles(data, out, all_caps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate RP test data from evaluation output.')
    parser.add_argument('-p', dest='path', type=str, help='Path to image directory')
    parser.add_argument('-c', dest="cap", default='all_texts.txt',
                        help='Optional: specify all texts file. Default to all texts in dataset.')
    parser.add_argument('-d', dest='out', type=str, help='Output directory', default='./')
    parser.add_argument('-t', dest='text', type=str, help='Directory to text data',
                        default='D:\\Projects\\Projects\\pytorch_Projects\\GAN\\Datasets\\attnGAN_data\\birds\\text')
    parser.add_argument('-r, --random', dest='rand', action='store_true',
                        help='If set, sample is selected randomly instead of sequentially.')

    args = parser.parse_args()

    args.p = 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\attnGAN_Results\\bird_AttnGAN2\\valid\\single'
    args.out = 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\attnGAN_Results\\bird_AttnGAN2\\RP_data\\'

    # # build main .txt file
    # pkl_path = 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\Datasets\\attnGAN_data\\birds\\captions.pickle'
    # all_sentences = read_all_from_pickle(pkl_path)
    # # print(123)

    # load text file
    if not os.path.isfile(args.cap):
        if args.cap == 'all_texts.txt':
            all_sentences = []
            # get total text
            for subDir in os.listdir(args.text):
                all_sentences += read_all_txt_orig(os.path.join(args.text, subDir))

            with open("all_texts.txt", mode='w') as f:
                f.writelines(all_sentences)
        else:
            print("Specify a valid text file containing all captions, or leave blank to use all texts.")
            quit()

    print(f"All text file: {args.cap}")
    #
    print("Processing {}.\nSample from its subfolders.".format(args.p))
    subDirs = []
    for each in os.listdir(args.p):
        full_path = os.path.join(args.p, each)
        if os.path.isdir(full_path):
            subDirs.append(full_path)

    main(subDirs, args.out, args.rand, args.cap)
