import os 
import torch.utils.data 
import torchvision.transforms as transforms
import numpy as np 
import json 
import torch 
import pickle
from PIL import Image
from torch.autograd import Variable 


class polyvore_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, meta_data, transform = None):
        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        self.impath = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        self.is_train = split == 'train'
        self.transform = transform
        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))

        imnames = set()
        self.id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                self.id2im['%s_%i' % (outfit_id, item['index'])] = im
                imnames.add(im)
        
        imnames = list(imnames)

        self.im2desc = {}            
        for im in imnames: 
            desc = meta_data[im]['title']
            if not desc:
                desc = meta_data[im]['url_name']
            
            desc = desc.replace('\n','').strip().lower()
            self.im2desc[im] = desc
        
        # load fill_in_the_blank tran/valid/test
        fitb_dir = os.path.join(rootdir, 'fill_in_blank_%s.json' % split)
        fitb = json.load(open(fitb_dir, 'r'))

        self.fitb_outfit_list = []
        self.target = []
        for line in fitb:
            temp = {}
            question = []
            answers = []
            for i in line['question']:
                question.append(self.id2im[i])
            temp['question'] = question
            for i in line['answers']:
                answers.append(self.id2im[i])
            temp['answers'] = answers
            temp['position'] = line['blank_position'] -1

            self.fitb_outfit_list.append(temp)
            outfit_id = line['question'][0].split('_')[0]
            answer_fit_id = [line['answers'][i].split('_')[0] for i in range(4)]
            self.target.append(answer_fit_id.index(outfit_id))

    
    def get_all_texts(self):
        if self.is_train:
            texts = list(self.im2desc.values())
        else:
            texts = []
        return texts

    def __len__(self):
        return len(self.target)
        
    
    def __getitem__(self, index):

        out = {'question_img': [],
               'question_text': [],
               'answers_img': [],
               'answers_text': [],
               'position': [],
               'target': []
        }

        fitb = self.fitb_outfit_list[index]
        for im in fitb['question']:
            img_path = os.path.join(self.impath, '%s.jpg' % im)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                out['question_img'].append(img)
            out['question_text'].append(self.im2desc[im])
        
        for im in fitb['answers']:
            img_path = os.path.join(self.impath, '%s.jpg' % im)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                out['answers_img'].append(img)
            out['answers_text'].append(self.im2desc[im])
        
        out['position'].append(fitb['position'])
        out['target'].append(self.target[index])

        return out
            



