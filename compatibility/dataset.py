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
        
        # load compatibility_train/valid/text
        compatibility_dir = os.path.join(rootdir, 'compatibility_%s.txt' % split)
        with open(compatibility_dir, 'r') as f:
            lines = f.readlines()
         
        self.outfit_list = []
        self.target = []
        for line in lines:
            outfit = []
            data = line.strip().split()
            for imid in data[1:]:
                outfit.append(self.id2im[imid])
            self.outfit_list.append(outfit)
            self.target.append(int(data[0]))
    
    def get_all_texts(self):
        if self.is_train:
            texts = list(self.im2desc.values())
        else:
            texts = []
        return texts

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        out = {'img':[],
               'target':[],
               'text':[]}
        for im in self.outfit_list[index]:
            img_path = os.path.join(self.impath, '%s.jpg' % im)
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                out['img'].append(img)
            out['text'].append(self.im2desc[im])
        out['target'].append(self.target[index])
        return out 

