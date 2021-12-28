import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import time
from HED import HED
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class HED_test():
    def __init__(self, cfg, weight_path):

        self.cfg = self.cfg_checker(cfg)


        mean = [float(item) / 255.0 for item in cfg.DATA.mean]
        std = [1,1,1]
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std) ])



        self.model = HED(self.cfg, None) 
        self.model = self.model.cuda()

        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        


    def cfg_checker(self, cfg):
        return cfg

        
    def extract_edge(self, img_raw):
        #img = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        img = img.unsqueeze(0)
        data = img.cuda()


        dsn1, dsn2, dsn3, dsn4, dsn5, dsn6 = self.model( data )  

        #save_img(dsn1, result_dir, 1)  
        #save_img(dsn2, result_dir, 2)  
        #save_img(dsn3, result_dir, 3)  
        #save_img(dsn4, result_dir, 4)  
        #save_img(dsn5, result_dir, 5)  
        #save_img(dsn6, result_dir, 6)  
        #pdb.set_trace()
        #if self.cfg.MODEL.loss_func_logits:
        #    dsn1 = torch.sigmoid(dsn1)
        #    dsn2 = torch.sigmoid(dsn2)
        #    dsn3 = torch.sigmoid(dsn3)
        #    dsn4 = torch.sigmoid(dsn4)
        #    dsn5 = torch.sigmoid(dsn5)
        #    dsn6 = torch.sigmoid(dsn6)

        #dsn7 = (dsn1 + dsn2 + dsn3 + dsn4 + dsn5) / 5.0
        #results = [dsn1,dsn2,dsn3,dsn4,dsn5,dsn6,dsn7]
        #for i in range(len(results)):
        #    each_dsn = results[i].data.cpu().numpy()
        #    each_dsn = np.squeeze( each_dsn )
        #    results[i] = each_dsn / np.max(each_dsn)
            
        if self.cfg.MODEL.loss_func_logits:
            dsn6 = torch.sigmoid(dsn6)

        each_dsn = dsn6.data.cpu().numpy()
        each_dsn = np.squeeze( each_dsn )
        result = each_dsn / np.max(each_dsn)

        return result


