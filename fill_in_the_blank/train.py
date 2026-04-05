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
parser.add_argument('--polyvore_split', default='disjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epoch_num', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--img_dim', type=int, default=256)
parser.add_argument('--word_embed_dim', type=int, default=512)
parser.add_argument('--max_outfit', type=int, default=10)
parser.add_argument('--model_dir', type=str, default='./results')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

def load_dataset(args):
    fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r'))
    trainset = polyvore_dataset(args, split='train', meta_data=meta_data, 
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.RandomCrop(224),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))
    validset = polyvore_dataset(args, split='valid', meta_data=meta_data, 
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))
    testset = polyvore_dataset(args, split='test', meta_data=meta_data, 
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(256),
                                torchvision.transforms.CenterCrop(224),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                            ]))
    print('trainset size:', len(trainset))
    print('valid size:', len(validset))
    print('testset size:', len(testset))
    return trainset, validset, testset


def create_model_and_optimizer(all_texts):
    """Builds the model and related optimizer."""

    img_model_ = img_model.Image_net(texts_to_build_vocab=all_texts,embedding_size=args.img_dim).cuda()
    img_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, img_model_.parameters()),lr = args.lr, eps=args.eps, weight_decay=args.weight_decay)
    text_model_ = text_model.Text_net(texts_to_build_vocab=all_texts, embedding_size=args.img_dim).cuda()
    text_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, text_model_.parameters()), lr = args.lr, eps=args.eps, weight_decay=args.weight_decay)
    return img_model_, text_model_, img_optimizer, text_optimizer

def compute_kl(x1, x2):
    log_soft_x1 = F.log_softmax(x1, dim=1)
    soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
    kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
    return kl

def compute_mse(x1, x2):
    return F.mse_loss(x1, x2)

def compute_auc_acc(predicted, label):

    fpr, tpr, thresholds = metrics.roc_curve(label, predicted, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)

    acc_predicted_score = predicted.squeeze()
    ones = np.ones_like(acc_predicted_score)
    zeros = np.zeros_like(acc_predicted_score)
    acc_predicted_score = np.where(acc_predicted_score > 0.5, ones, zeros)

    acc_score = metrics.accuracy_score(label, acc_predicted_score)
    return auc_score, acc_score

def compute_fitb_acc(predicted, label):
    predicted_max_index = np.argmax(predicted, axis=1)
    total_num = predicted.shape[0]
    correct_num = np.where(predicted_max_index == label, 1, 0)
    return np.sum(correct_num) / float(total_num)

def train(img_model, text_model, img_optimizer, text_optimizer, dataloader, scaler):
    img_model.train()
    text_model.train()

    img_all_predicted_socre = []
    text_all_predicted_score = []
    all_labels = []

    with tqdm(total = 2*len(dataloader)) as t:
        for i,data in enumerate(dataloader):

            question_img = [x['question_img'] for x in data]
            question_text = [x['question_text'] for x in data]
            answers_img = [x['answers_img'] for x in data]
            answers_text = [x['answers_text'] for x in data]
            position = [x['position'] for x in data]
            target = [x['target'] for x in data]
        
            labels = torch.tensor(target).squeeze(1).cuda()
            # optimize img
            img_optimizer.zero_grad()

            with autocast():
                img_predicted_score, loss_consist, loss_ortho = img_model(question_img, question_text, answers_img, answers_text, position)
                text_predicted_score = text_model(question_img, question_text, answers_img, answers_text, position)[0].detach()
                img_classify_loss = F.cross_entropy(img_predicted_score, labels)
                img_mutual_loss = compute_kl(img_predicted_score, text_predicted_score)
                img_total_loss = img_classify_loss + img_mutual_loss + loss_consist + loss_ortho
            scaler.scale(img_total_loss).backward()
            scaler.step(img_optimizer)
            scaler.update()
            predicted_y = F.softmax(img_predicted_score, dim=1)
            t.set_postfix(img_classify_loss='{:05.3f}'.format(img_classify_loss.item()), img_mutual_loss='{:05.3f}'.format(img_mutual_loss.item()), loss_consist='{:05.3f}'.format(loss_consist.item()), loss_ortho='{:05.3f}'.format(loss_ortho.item()))
            t.update()

            #optimize text
            text_optimizer.zero_grad()
            with autocast():
                text_predicted_score, loss_consist, loss_ortho = text_model(question_img, question_text, answers_img, answers_text, position)
                img_predicted_score = img_model(question_img, question_text, answers_img, answers_text, position)[0].detach()
                text_classify_loss = F.cross_entropy(text_predicted_score, labels)
                text_mutual_loss = compute_kl(text_predicted_score, img_predicted_score)
                text_total_loss = text_classify_loss + text_mutual_loss + loss_consist + loss_ortho
            scaler.scale(text_total_loss).backward()
            scaler.step(text_optimizer)
            scaler.update()

            predicted_y = F.softmax(text_predicted_score, dim=1)
            t.set_postfix(text_classify_loss='{:05.3f}'.format(text_classify_loss.item()), text_mutual_loss='{:05.3f}'.format(text_mutual_loss.item()), loss_consist='{:05.3f}'.format(loss_consist.item()), loss_ortho='{:05.3f}'.format(loss_ortho.item()))
            t.update()

def test(img_model, text_model, dataloader):
    img_model.eval()
    text_model.eval()
    img_all_predicted_socre = []
    text_all_predicted_score = []
    all_labels = []
    with torch.no_grad():
        with tqdm(total = len(dataloader)) as t:
            for i,data in enumerate(dataloader):

                question_img = [x['question_img'] for x in data]
                question_text = [x['question_text'] for x in data]
                answers_img = [x['answers_img'] for x in data]
                answers_text = [x['answers_text'] for x in data]
                position = [x['position'] for x in data]
                target = [x['target'] for x in data]

                img_predicted_score = img_model(question_img, question_text, answers_img, answers_text, position)[0]
                text_predicted_score = text_model(question_img, question_text, answers_img, answers_text, position)[0]

                labels = torch.tensor(target).squeeze(1).cuda()
                predicted_y = F.softmax(img_predicted_score, dim=1)
                img_all_predicted_socre += [predicted_y.data.cpu().numpy()]
                all_labels += [labels.data.cpu().numpy()]
                predicted_y = F.softmax(text_predicted_score, dim=1)
                text_all_predicted_score += [predicted_y.data.cpu().numpy()]
                t.update()

    img_all_predicted_socre = np.concatenate(img_all_predicted_socre)
    text_all_predicted_score = np.concatenate(text_all_predicted_score)
    all_labels = np.concatenate(all_labels)

    acc = compute_fitb_acc((img_all_predicted_socre+text_all_predicted_score)/2.0, all_labels)

    return acc

def train_and_evaluate(img_model, text_model, img_optimizer, text_optimizer, trainset, validset, testset):
    trainloader = dataloader.DataLoader(trainset, 
                             batch_size = args.batch_size,
                             shuffle = True,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn=lambda i:i)
    validloader = dataloader.DataLoader(validset, 
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn = lambda i:i)      
    testloader = dataloader.DataLoader(testset, 
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = True,
                             num_workers=args.num_workers,
                             collate_fn= lambda i :i)    
    scaler = GradScaler()

    best_avg_score = float('-inf')
    for epoch in range(args.epoch_num):

        logging.info("Epoch {}/{}".format(epoch + 1, args.epoch_num))
        train(img_model, text_model, img_optimizer, text_optimizer, trainloader, scaler)

        valid_acc = test(img_model,text_model, validloader)
        logging.info("valid acc {}".format(valid_acc))

        if valid_acc > best_avg_score:
            best_avg_score = valid_acc
            test_acc = test(img_model,text_model, testloader)
            logging.info("Fine new avg best score at epoch {}, test_acc {}".format(epoch, test_acc))
            best_test_saved_avg = os.path.join(args.model_dir, "metrics_best_avg.txt")
            with open(best_test_saved_avg, 'w') as f:
                f.write('%s: test_acc: %s ' % (str(epoch), str(test_acc)))
            torch.save(img_model, os.path.join(args.model_dir, 'img_model.pt'))
            torch.save(text_model, os.path.join(args.model_dir, 'text_model.pt'))


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
    # fetch dataloaders

    trainset, validset, testset = load_dataset(args)
    img_model, text_model, img_optimizer, text_optimizer = create_model_and_optimizer(all_texts=trainset.get_all_texts())

    logging.info('- done.')

    # Train the model
    logging.info("Starting train for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(img_model, text_model, img_optimizer, text_optimizer, trainset, validset, testset)
