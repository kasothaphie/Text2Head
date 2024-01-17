import torch
from torch import Tensor
from typing import List, Optional

class EMA(torch.optim.Optimizer):
    def __init__(self, params, beta=0.9, lr=1e-3, weight_decay=0, maximize=False):
        defaults = dict(beta=beta, lr=lr, weight_decay=weight_decay, maximize=maximize)
        super(EMA, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            beta = group['beta']
            lr = group['lr']
            weight_decay = group['weight_decay']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data if not maximize else -p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential MA of gradients
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']

                state['step'] += 1

                # Add weight decay if any
                if weight_decay != 0:
                    grad = grad.add(weight_decay, p.data)

                # EMA
                exp_avg.mul_(beta).add_((1 - beta), grad)

                p.data = p.data - lr * exp_avg

                # Save state
                state['exp_avg'] = exp_avg
        return loss
