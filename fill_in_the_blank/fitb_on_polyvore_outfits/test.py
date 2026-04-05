import argparse
import os
import sys
import json
import logging
import warnings
from tqdm import tqdm

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from dataset import polyvore_dataset
from torch.utils.data import dataloader
import img_model 
import text_model

from torch.cuda.amp import autocast as autocast, GradScaler

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(4)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Fashion Compatibility Example')
parser.add_argument('--datadir', default='./', type=str,
                    help='directory of the polyvore outfits dataset')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                    help='specifies the split of the polyvore data')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epoch_num', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 5e-5)')

parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-6)


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--img_dim', type=int, default=256)
parser.add_argument('--word_embed_dim', type=int, default=512)
parser.add_argument('--max_outfit', type=int, default=10)

parser.add_argument('--model_dir', type=str, default='./experiment')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

def compute_fitb_acc(predicted, label):
    predicted_max_index = np.argmax(predicted, axis=0)
    total_num = predicted.shape[1]
    correct_num = np.where(predicted_max_index == label, 1, 0)
    return np.sum(correct_num) / float(total_num)


def test():
    text_model = torch.load('./text_model.pt')
    img_model = torch.load('./img_model.pt')
    text_model.eval()
    img_model.eval()
    fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    testset = polyvore_dataset(args, split='test', meta_data=meta_data, 
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))

    testloader = dataloader.DataLoader(testset, 
                            batch_size = args.batch_size,
                            shuffle = False,
                            drop_last = True,
                            num_workers=args.num_workers,
                            collate_fn= lambda i :i)    

    img_all_predicted_score = [[],[],[],[]]
    text_all_predicted_score = [[],[],[],[]]
    all_labels = []

    with torch.no_grad():
        with tqdm(total = 4 * len(testloader)) as t:
            for i,data in enumerate(testloader):
                data0 = [t[0] for t in data]
                data1 = [t[1] for t in data]
                data2 = [t[2] for t in data]
                data3 = [t[3] for t in data]

                data_current = [data0, data1, data2, data3]
                for _, d in enumerate(data_current):
                    img = [t['img'] for t in d]
                    text = [t['text'] for t in d]
                    target = [t['target'] for t in d]

                    img_predicted_score = img_model(img, text)[0]
                    text_predicted_score = text_model(text, img)[0]
                    labels = torch.tensor(target).squeeze(1).cuda()

                    predicted_y = F.softmax(img_predicted_score, dim=1)
                    predicted_y = predicted_y[:,1]
                    img_all_predicted_score[_] += [predicted_y.data.cpu().numpy()]
                    if _ == 0:
                        all_labels += [labels.data.cpu().numpy()]

                    predicted_y = F.softmax(text_predicted_score, dim=1)
                    predicted_y = predicted_y[:,1]
                    text_all_predicted_score[_] += [predicted_y.data.cpu().numpy()]
                    t.update()

    img_all_predicted_score = [np.concatenate(i) for i in img_all_predicted_score]    
    img_all_predicted_score = np.stack(img_all_predicted_score)

    text_all_predicted_score = [np.concatenate(i) for i in text_all_predicted_score]    
    text_all_predicted_score = np.stack(text_all_predicted_score)

    all_labels = np.concatenate(all_labels)
    acc = compute_fitb_acc(img_all_predicted_score + text_all_predicted_score, all_labels)

    print(acc)




if __name__ == '__main__':

    # Load the parameters from json file

    print('Arguments:')
    for k in args.__dict__.keys():
        print('    ', k, ':', str(args.__dict__[k]))
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(os.path.join(args.model_dir, 'train.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info('Loading the datasets and model...')
    test()
