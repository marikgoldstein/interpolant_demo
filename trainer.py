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

# local
from unet import Unet
from ode_int import PFlowIntegrator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False


def setup_wandb(config):
    if not config.use_wandb:
        return

    config.wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            resume=None,
            id    =None,
    )

    config.wandb_run_id = config.wandb_run.id

    for key in vars(config):
        item = getattr(config, key)
        if is_type_for_logging(item):
            setattr(wandb.config, key, item)
    print("finished wandb setup")

# Wrapper around model
class Velocity(nn.Module):
    def __init__(self, config):
        
        super(Velocity, self).__init__()
        self.config = config
        self._arch = Unet(
            num_classes = config.num_classes,
            in_channels = config.C,
            out_channels= config.C,
            dim = config.unet_channels,
            dim_mults = config.unet_dim_mults,
            resnet_block_groups = config.unet_resnet_block_groups,
            learned_sinusoidal_cond = config.unet_learned_sinusoidal_cond,
            random_fourier_features = config.unet_random_fourier_features,
            learned_sinusoidal_dim = config.unet_learned_sinusoidal_dim,
            attn_dim_head = config.unet_attn_dim_head,
            attn_heads = config.unet_attn_heads,
            use_classes = config.unet_use_classes,
        )
        num_params = np.sum([int(np.prod(p.shape)) for p in self._arch.parameters()])
        print("Num params in main arch for velocity is", f"{num_params:,}")

    def forward(self, zt, t, y, cond=None):
        
        if not self.config.unet_use_classes:
            y = None

        return self._arch(zt, t, y)

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
        self.model = Velocity(self.config)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.base_lr)
        self.current_epoch = 0
        self.global_step = 0
        self.loader = get_dataloader(config) 
        self.pflow = PFlowIntegrator(config)
        self.overfit_batch  = next(iter(self.loader)) 
        self.time_dist = torch.distributions.Uniform(low=self.config.t_min_train, high=self.config.t_max_train)
        setup_wandb(self.config)
        self.print_config()

    def print_config(self,):
        c = self.config
        for key in vars(c):
            val = getattr(c, key)
            if is_type_for_logging(val):
                print(key, val)

    def wide(self, x):
        return x[:, None, None, None]

    def alpha(self, t):
        return 1-t

    def alpha_dot(self, t):
        return -1.0 * torch.ones_like(t)
    
    def beta(self, t):
        return t

    def beta_dot(self, t):
        return 1.0 * torch.ones_like(t)

    def loss_target(self, D):
        aterm = self.wide(self.alpha_dot(D['t'])) * D['z0'] 
        bterm = self.wide(self.beta_dot(D['t'])) * D['z1']
        return aterm + bterm

    def It(self, D):
        aterm = self.wide(self.alpha(D['t'])) * D['z0']
        bterm = self.wide(self.beta(D['t'])) * D['z1']
        D['zt'] = aterm + bterm 
        return D

    def get_time(self, D):
        t = self.time_dist.sample(sample_shape = (D['z1'].shape[0],))
        D['t'] = t.squeeze().type_as(D['z1'])
        return D

    def center(self, x):
        return (x * 2.0) - 1.0

    @torch.no_grad()
    def prepare_batch(self, batch, force_overfit_batch = False):

        if (batch is None) or (force_overfit_batch):
            batch = self.overfit_batch

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        z1 = (x * 2.0) - 1.0

        D = {'N': z1.shape[0], 'y': y, 'z1': z1}

        D = self.get_time(D)

        D['z0'] = torch.randn_like(D['z1'])

        D = self.It(D)
         
        return D

    def odeint(self, data_dict):
        # steps = 5 is just a placeholder
        # dopri5 is adaptive and doesn't look at steps arg.
        # (but even then, steps has to be >= 3 not to cause a bug)
        return self.pflow(
            b = self.model,
            z0 = data_dict['z0'],
            y = data_dict['y'],
            T_min = self.config.t_min_sample,
            T_max = self.config.t_max_sample,
            steps = 5, 
            method = 'dopri5',
            return_last = True,
        )
    def maybe_log_wandb(self, sample, D, normalize = True):
        
        if not self.config.use_wandb:
            return

        sample = to_grid(sample, normalize = normalize)
        
        z1 = to_grid(D['z1'], normalize = normalize)
        z0 = to_grid(D['z0'], normalize = normalize)
       
        # plot (z0, sample, z1)
        everything = torch.cat([z0, sample, z1], dim=-1)
       
        everything = wandb.Image(everything)
        wandb.log({'z0,sample,z1': everything}, step = self.global_step)

    @torch.no_grad()
    def definitely_sample(self,):
        self.model.eval()
        data_dict = self.prepare_batch(batch = None, force_overfit_batch = True)
        zT = self.odeint(data_dict)
        self.maybe_log_wandb(zT, data_dict, 'samples')           

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

    def training_step(self, D):
        assert self.model.training
        model_out = self.model(D['zt'], D['t'], D['y'])
        target = self.loss_target(D)
        return self.image_norm(model_out - target).mean()

    def fit(self,):

        self.definitely_sample()
        print("starting training")
        while self.global_step < self.config.max_steps:


            print(f"starting epoch {self.current_epoch}")

            for batch_idx, batch in enumerate(self.loader):
                           
                if self.global_step >= self.config.max_steps:
                    break

                data_dict = self.prepare_batch(batch, force_overfit_batch = self.config.overfit)
                self.model.train()
                loss = self.training_step(D = data_dict)
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

        self.integration_atol = 1.e-5
        self.integration_rtol = 1.e-5

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

