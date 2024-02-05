import torch
import torch.nn as nn
from torch.nn import functional as F

class MonoNPHM(nn.Module):

    def __init__(self, GTA, latent_code, latent_codes_expr):
        super().__init__()
        self.GTA = GTA

        self.latent_code_id = latent_code
        self.latent_codes_expr = latent_codes_expr


    def forward(self, positions, expression, include_color : bool = False, return_grad : bool = False):
        condition = self.latent_code_id
        condition.update({'exp': self.latent_codes_expr(torch.tensor([expression], device=positions.device)).unsqueeze(0)})
        if len(positions.shape) == 2:
            positions = positions.unsqueeze(0)
        result = self.GTA({'queries': positions}, cond=condition, skip_color=not include_color, return_grad=return_grad)
        sdf = result['sdf'].squeeze(0)

        if include_color:
            color = result['color'].squeeze(0)
            #sdf, _, color = self.GTA(positions, self.latent_code, None, squeeze=True)
            if return_grad:
                return torch.cat([sdf, color, result['gradient'].squeeze(0)], dim=-1)
            else:
                return torch.cat([sdf, color], dim=-1)
        else:
            return sdf

    def warp(self, positions, expression):
        condition = self.latent_code_id
        condition.update({'exp': self.latent_codes_expr(torch.tensor([expression], device=positions.device)).unsqueeze(0)})

        in_dict = {'queries': positions, 'cond': condition}

        if hasattr(self.GTA.id_model, 'mlp_pos') and self.GTA.id_model.mlp_pos is not None and 'anchors' not in in_dict:
            in_dict.update({'anchors': self.GTA.id_model.get_anchors(condition['geo'])})
        out_ex = self.GTA.ex_model(in_dict)
        queries_canonical = in_dict['queries'] + out_ex['offsets']


        return queries_canonical

    def gradient(self, x, expression):
        x.requires_grad_(True)
        condition = self.latent_code_id
        condition.update({'exp': self.latent_codes_expr(torch.tensor([expression], device=x.device)).unsqueeze(0)})
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        result = self.GTA.forward({'queries': x}, condition, None)
        y = result['sdf']
        y = y.squeeze(0)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0].squeeze(0)
        return gradients.unsqueeze(1)
