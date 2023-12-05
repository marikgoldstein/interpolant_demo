# USES TORCHDIFFEQ https://github.com/rtqichen/torchdiffeq

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint #odeint_adjoint as odeint
import math


class PFlowRHS(nn.Module):
    def __init__(self, config, b, y):
        super(PFlowRHS, self).__init__()
        self.config = config
        self.b = b
        self.y = y
        
    def forward(self, t, states):
        (zt,) = states
        t_arr = torch.ones(zt.shape[0]).type_as(zt) * t
        dzt = self.b(zt = zt, t = t_arr, y = self.y)
        return (dzt,)
             
class PFlowIntegrator:
        
    def __init__(self, config):
        
        self.config = config

    def __call__(self, b, z0, y, T_min, T_max, steps, method='dopri5', return_last = True):

        c = self.config
        
        rhs = PFlowRHS(config = c, 
            b = b,
            y = y, 
        )

        t = torch.linspace(
            T_min, T_max, steps
        ).type_as(z0)

        int_args = {
            'method': method, 
            'atol': c.integration_atol, 
            'rtol': c.integration_rtol,
        }

        (z,) = odeint(rhs, (z0,), t, **int_args)
        if return_last:
            return z[-1]
        else:
            return z

