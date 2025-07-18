import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader 
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from PIL import Image
from dataclasses import dataclass

# local
import unet

@dataclass
class Batch:
    x0:torch.Tensor
    x1:torch.Tensor
    t:torch.Tensor
    y:torch.Tensor
    xt:torch.Tensor


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

def wide(t):
    return t[:, None, None, None]

def setup_wandb(config):
    
    if not config.use_wandb:
        return

    config.wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            resume=None,
            id    =None,
    )


# Wrapper around model
class Model(nn.Module):
    def __init__(self, config):
        
        super(Model, self).__init__()
        self.config = config
        self._arch = unet.get_unet(config)
        num_params = np.sum([int(np.prod(p.shape)) for p in self._arch.parameters()])
        print("Num params in main arch for velocity is", f"{num_params:,}")

    def forward(self, xt, t, y):
        
        if not self.config.unet_use_classes:
            y = None

        return self._arch(xt, t, y)


def to_grid(x, normalize):
    nrow = int(np.floor(np.sqrt(x.shape[0])))
    if normalize:
        kwargs = {'normalize' : True, 'value_range' : (-1, 1)}
    else:
        kwargs = {}
    return make_grid(x, nrow = nrow, **kwargs)

def clip_grad_norm(model):
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm = 10000, norm_type= 2.0, error_if_nonfinite = False
    )

def get_dataloader(config):
    Flip = T.RandomHorizontalFlip()
    Tens = T.ToTensor()
    transform = T.Compose([Flip, Tens])
    ds = datasets.CIFAR10(config.data_path, train=True, download=True, transform=transform)
    return DataLoader(
        ds,
        batch_size = config.batch_size,
        shuffle = True, 
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = True, 
    )

class Trainer:
    
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(self.config)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.base_lr)
        self.current_epoch = 0
        self.global_step = 0
        self.loader = get_dataloader(config) 
        self.overfit_batch  = next(iter(self.loader)) 
        self.time_dist = torch.distributions.Uniform(low=self.config.t_min_train, high=self.config.t_max_train)
        setup_wandb(self.config)

    def center(self, x):
        return (x * 2.0) - 1.0

    @torch.no_grad()
    def prepare_batch(self, batch, overfit=False):

        if overfit:
            batch = self.overfit_batch

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        N = x.shape[0]
        # center
        x1 = self.center(x)

        t = self.time_dist.sample(sample_shape = (N,)).type_as(x1)

        x0 = torch.randn_like(x1)

        xt = wide(1-t) * x0 + wide(t) * x1
        
        return Batch(x0=x0,x1=x1,xt=xt,y=y,t=t)


    def maybe_log_wandb(self, sample):
        
        if not self.config.use_wandb:
            return

        sample = to_grid(sample, normalize = True)
        sample = wandb.Image(sample)
        wandb.log({'sample': sample}, step=self.global_step)

    @torch.no_grad()
    def definitely_sample(self,):
        self.model.eval()
        batch = self.prepare_batch(batch=None, overfit=True)
        steps = self.config.sample_steps
       
        xt = batch.x0
        N = xt.shape[0]
        times = torch.linspace(
            self.config.t_min_sample, self.config.t_max_sample, steps
        ).to(xt.device)
        dt = times[1] - times[0]
        ones = torch.ones(N,).to(xt.device)
        for t_scalar in times:
            tvec = t_scalar * ones
            xt = xt + dt * self.model(xt, tvec, batch.y)
        x1_hat = xt
        self.maybe_log_wandb(x1_hat)

    @torch.no_grad()
    def maybe_sample(self,):
        if self.global_step % self.config.sample_every == 0:
            self.definitely_sample()
    
    def optimizer_step(self,):
        clip_grad_norm(self.model)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
    
    def image_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def training_step(self, batch):
        assert self.model.training
        model_out = self.model(batch.xt, batch.t, batch.y)
        target = batch.x1 - batch.x0
        return self.image_norm(model_out - target).mean()

    def fit(self,):

        self.definitely_sample()
        print("starting training")
        while self.global_step < self.config.max_steps:

            print(f"starting epoch {self.current_epoch}")

            for batch_idx, batch in enumerate(self.loader):
                           
                if self.global_step >= self.config.max_steps:
                    break

                batch = self.prepare_batch(batch, overfit = self.config.overfit)
                self.model.train()
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer_step()            
                self.maybe_sample()

                if self.global_step % self.config.print_loss_every == 0:
                    print(f"Grad step {self.global_step}. Loss:{loss.item()}")

            self.current_epoch += 1

class Config:
    def __init__(self,):
        self.H = 32
        self.W = 32
        self.C = 3
        self.num_classes = 10
        self.data_path = '../data/'
        self.batch_size = 128
        self.num_workers = 4

        self.t_min_train = 0.001
        self.t_max_train = 1 - 0.001
        
        self.t_min_sample = 0.001
        self.t_max_sample = 1 - 0.001
        self.sample_steps = 100

        self.use_wandb = True 
        self.wandb_project = 'columbia'
        self.wandb_entity = 'marikgoldstein'

        self.overfit = True
        if self.overfit:
            print("NOTE: In overfit mode")

        self.sample_every = 100
        self.print_loss_every = 20

        print(f"Printing loss every {self.print_loss_every}")
        print(f"Sampling every {self.sample_every}")
        
        # some training hparams
        self.base_lr = 2e-4 
        self.max_steps = 1_000_000
        
        self.unet_use_classes = True
        self.unet_channels = 128
        self.unet_dim_mults = (1, 2, 2, 2)
        self.unet_resnet_block_groups = 8
        self.unet_learned_sinusoidal_dim = 32
        self.unet_attn_dim_head = 64
        self.unet_attn_heads = 4
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False

def main():

    conf = Config()
    trainer = Trainer(conf)
    trainer.fit()

if __name__ == '__main__':
    main() 

