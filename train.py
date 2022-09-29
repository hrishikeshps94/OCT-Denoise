# import sys
# sys.path.append('model.py')
from tabnanny import verbose
from model import UNet
import torch,os
from utils import Masker
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import DenoiseDataset,ValDenoiseDataset
from torch.utils.data import DataLoader
import argparse
import tqdm
import numpy as np
from metrics import calculate_ssim,calculate_psnr
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50'])
parser.add_argument('--resume', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--data_dir', type=str,
                    default='PKU37_OCT_Denoising/ds/train')
parser.add_argument('--val_dirs', type=str, default='PKU37_OCT_Denoising/ds/val')
parser.add_argument('--subfold', type=str, default='OCTA')
parser.add_argument('--save_model_path', type=str,
                    default='experiments/OCTA')
parser.add_argument('--log_name', type=str,
                    default='octa')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--w_decay', type=float, default=1e-8)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=12)
parser.add_argument('--patchsize', type=int, default=128)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=2.0)
parser.add_argument("--increase_ratio", type=float, default=20.0)

opt, _ = parser.parse_known_args()

class Train():
    def __init__(self,opt) -> None:
        self.opt = opt
        self.Thread1 = 0.4
        self.Thread2 = 1.0
        self.Lambda1 = opt.Lambda1
        self.Lambda2 = opt.Lambda2
        self.increase_ratio = opt.increase_ratio
        self.dataset_generator()
        self.model_intialisation()
        self.optimiser_loss_intialiser()
        self.best_psnr_dn = 0
        self.best_psnr_exp = 0
        self.best_psnr_mid = 0


    def model_intialisation(self):
        model = UNet(in_channels=self.opt.n_channel,
                out_channels=self.opt.n_channel,
                wf=self.opt.n_feature)
        self.masker = Masker(width=4, mode='interpolate', mask_type='all')
        self.model = model.to(device)
        # self.model.load_state_dict(torch.load('best.path'))
    def optimiser_loss_intialiser(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr,
                       weight_decay=self.opt.w_decay)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=(len(self.train_dataloader)*self.opt.n_epochs))
        print("Batchsize={}, number of epoch={}".format(self.opt.batchsize, self.opt.n_epochs))
    def dataset_generator(self):
        train_data = DenoiseDataset(data_dir=self.opt.data_dir,patch=self.opt.patchsize)
        self.train_dataloader = DataLoader(dataset=train_data,num_workers=os.cpu_count(),batch_size=self.opt.batchsize,shuffle=True,pin_memory=False,drop_last=True)
        val_data = ValDenoiseDataset(data_dir=self.opt.val_dirs)
        self.val_dataloader = DataLoader(dataset=val_data,num_workers=os.cpu_count(),batch_size=1,shuffle=True,pin_memory=False,drop_last=False)

    def train_one_epoch(self):
        self.model.train()
        total_loss_all = 0
        total_loss_rev = 0
        total_loss_reg = 0
        total_iterations = len(self.train_dataloader)
        for iteration, noisy in enumerate(tqdm.tqdm((self.train_dataloader))):
            noisy = noisy.to(device)
            self.optimizer.zero_grad()
            net_input, mask = self.masker.train(noisy)
            # with torch.set_grad_enabled(True):
            noisy_output = self.model(net_input)
            n, c, h, w = noisy.shape
            noisy_output = (noisy_output*mask).view(n, -1, c, h, w).sum(dim=1)
            diff = noisy_output - noisy
            with torch.set_grad_enabled(False):
                exp_output = self.model(noisy)
            exp_diff = exp_output - noisy
            Lambda = self.epoch / self.opt.n_epochs
            if Lambda <= self.Thread1:
                self.beta = self.Lambda2
            elif self.Thread1 <= Lambda <= self.Thread2:
                self.beta = self.Lambda2 + (Lambda - self.Thread1) * \
                    (self.increase_ratio-self.Lambda2) / (self.hread2-self.Thread1)
            else:
                self.beta = self.increase_ratio
            alpha = self.Lambda1
            revisible = diff + self.beta * exp_diff
            loss_reg = alpha * torch.mean(diff**2)
            loss_rev = torch.mean(revisible**2)
            loss_all = loss_reg + loss_rev
            total_loss_all+=loss_all.item()
            total_loss_reg+=loss_reg.item()
            total_loss_rev+=loss_rev.item()
            loss_all.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        print(
            '{:04d} / {:04d} Loss_Reg={:.6f},  Loss_Rev={:.6f}, Loss_All={:.6f},LR = {:0.6f}'
            .format(self.epoch,self.opt.n_epochs,total_loss_reg/total_iterations, total_loss_rev/total_iterations, total_loss_all/total_iterations,self.optimizer.param_groups[0]['lr']))
    def validate_one_epoch(self):
        avg_psnr_dn = []
        avg_ssim_dn = []
        avg_psnr_exp = []
        avg_ssim_exp = []
        avg_psnr_mid = []
        avg_ssim_mid = []
        for iteration,(noisy,noisy_255,origin255) in enumerate(tqdm.tqdm((self.val_dataloader))):
            noisy = noisy.to(device)
            with torch.no_grad():
                n, c, h, w = noisy.shape
                net_input, mask = self.masker.train(noisy)
                noisy_output = (self.model(net_input) *
                        mask).view(n, -1, c, h, w).sum(dim=1)
                dn_output = noisy_output.detach().clone()
                # Release gpu memory
                del net_input, mask, noisy_output
                torch.cuda.empty_cache()
                exp_output = self.model(noisy)
                pred_dn = dn_output
                pred_exp = exp_output.detach().clone()
                pred_mid = (pred_dn + self.beta*pred_exp) / (1 + self.beta)

                del exp_output
                torch.cuda.empty_cache()

                pred_dn = pred_dn.permute(0, 2, 3, 1)
                pred_exp = pred_exp.permute(0, 2, 3, 1)
                pred_mid = pred_mid.permute(0, 2, 3, 1)
                origin255 = origin255.permute(0, 2, 3, 1)
                noisy_255 = noisy_255.permute(0, 2, 3, 1)

                pred_dn = pred_dn.cpu().data.clamp(0, 1).numpy().squeeze(0)
                pred_exp = pred_exp.cpu().data.clamp(0, 1).numpy().squeeze(0)
                pred_mid = pred_mid.cpu().data.clamp(0, 1).numpy().squeeze(0)
                origin255 = origin255.cpu().numpy().squeeze(0)
                noisy_255 = noisy_255.cpu().numpy().squeeze(0)

                pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                        255).astype(np.uint8)
                pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                        255).astype(np.uint8)
                pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                        255).astype(np.uint8)

                # calculate psnr
                psnr_dn = calculate_psnr(origin255.astype(np.float32),
                                            pred255_dn.astype(np.float32))
                avg_psnr_dn.append(psnr_dn)
                ssim_dn = calculate_ssim(origin255.astype(np.float32),
                                            pred255_dn.astype(np.float32))
                avg_ssim_dn.append(ssim_dn)

                psnr_exp = calculate_psnr(origin255.astype(np.float32),
                                            pred255_exp.astype(np.float32))
                avg_psnr_exp.append(psnr_exp)
                ssim_exp = calculate_ssim(origin255.astype(np.float32),
                                            pred255_exp.astype(np.float32))
                avg_ssim_exp.append(ssim_exp)

                psnr_mid = calculate_psnr(origin255.astype(np.float32),
                                            pred255_mid.astype(np.float32))
                avg_psnr_mid.append(psnr_mid)
                ssim_mid = calculate_ssim(origin255.astype(np.float32),
                                            pred255_mid.astype(np.float32))
                avg_ssim_mid.append(ssim_mid)
        disp_im = np.concatenate([noisy_255,pred255_dn,origin255],axis=1)
        


        avg_psnr_dn = np.array(avg_psnr_dn)
        avg_psnr_dn = np.mean(avg_psnr_dn)
        avg_ssim_dn = np.mean(avg_ssim_dn)

        avg_psnr_exp = np.array(avg_psnr_exp)
        avg_psnr_exp = np.mean(avg_psnr_exp)
        avg_ssim_exp = np.mean(avg_ssim_exp)

        avg_psnr_mid = np.array(avg_psnr_mid)
        avg_psnr_mid = np.mean(avg_psnr_mid)
        avg_ssim_mid = np.mean(avg_ssim_mid)
        print("epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}\n".format(
                            self.epoch, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid))
        print("epoch:{},best_dn:{:.6f},best_exp:{:.6f},best_mid:{:.6f}\n".format(self.epoch,self.best_psnr_dn,\
            self.best_psnr_exp,self.best_psnr_mid))
        return avg_psnr_dn,avg_psnr_exp,avg_psnr_mid,disp_im

    def run(self):
        for self.epoch in range(self.opt.n_epochs):
            self.train_one_epoch()
            avg_psnr_dn,avg_psnr_exp,avg_psnr_mid,disp_im = self.validate_one_epoch()
            save_data = {
            'step': self.epoch,
            'best_psnr_dn':self.best_psnr_dn,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'beta':self.beta}
            if avg_psnr_dn>self.best_psnr_dn:
                torch.save(save_data,'best.pth')
                cv2.imwrite(f'Results/{self.epoch}.jpg',cv2.cvtColor(disp_im,cv2.COLOR_RGB2BGR))
                self.best_psnr_dn = avg_psnr_dn
            if avg_psnr_exp>self.best_psnr_exp:
                self.best_psnr_exp = avg_psnr_exp
            if avg_psnr_mid>self.best_psnr_mid:
                self.best_psnr_mid = avg_psnr_mid


trainer = Train(opt)
trainer.run()

            
