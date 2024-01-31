from typing import Callable, Optional, Tuple, Union
from time import time
import torch.nn
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor
import torch_geometric
from torch_geometric.utils import index_sort
import numpy as np


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear, HeteroLinear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse
)
from torch_geometric.utils import softmax

from pytorch3d.ops import knn_points, ball_query

from typing import Optional

from torch import Tensor

from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

from nphm_tum import env_paths




def softmax_mod_varySize2(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    num_neighbors: int = 8,
    dim: int = 0,
) -> Tensor:
    '''
    Implements the blending of local MLP predictions.
    Instead of using a Gaussian influence function that is normalized,
        the radius (i.e. the standard deviation) of the Gaussian is scaled based on the distance to the k-th nearest neighbor.
    :param src:
    :param index:
    :param ptr:
    :param num_nodes:
    :param num_neighbors:
    :param dim:
    :return:
    '''

    N = maybe_num_nodes(index, num_nodes)
    #src /= (1e-12 + torch.std(src.detach().view(-1, num_neighbors+1)[:, :num_neighbors], dim=1, keepdim=True).repeat(1, num_neighbors+1).view(-1))
    #src *= 2
    src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')

    #out = src - src_max.index_select(dim, index)
    #out = (-src/(0.1**2)).exp()
    local_std = 0.25**2
    dists = src.view(-1, num_neighbors + 1)[:, :num_neighbors]
    local_std = torch.max(dists, dim=-1, keepdim=False)[0] / 4
    local_std = torch.repeat_interleave(local_std, num_neighbors + 1)
    #local_std = torch.clamp(local_std, 0.05, max=None)
    out =(-(src/local_std).square()/2).exp() #/(2.5066*local_std)

    #out = torch.clamp(out, 0, 1)
    #out = torch.clamp(out - torch.min(out.view(-1, num_neighbors+1)[:, :num_neighbors], dim=1, keepdim=True)[0].repeat(1, num_neighbors+1).view(-1), min=0)
    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-8
    out_sum = out_sum.index_select(dim, index)


    return out / out_sum


class GlobalField(nn.Module):
    '''
    Implements SDF (and optionally the texture field) in canonical space as a simple MLP.
    '''
    def __init__(
            self,
            lat_dim,
            hidden_dim,
            nlayers=8,
            geometric_init=True,
            radius_init=1,
            beta=100,
            out_dim=1,
            num_freq_bands=None,
            input_dim=3,
            return_last_feats=False,
            color_branch=False,
            n_hyper : int = 0,
            sdf_corrective : bool = False,
            pass_pos_to_app_head : bool  = False,
            lat_dim_app : int = 0

    ):
        super().__init__()
        if num_freq_bands is None:
            d_in_spatial = input_dim
        else:
            d_in_spatial = input_dim*(2*num_freq_bands+1)
        d_in = lat_dim + d_in_spatial
        self.lat_dim = lat_dim
        self.lat_dim_app = lat_dim_app
        self.input_dim = input_dim
        self.color_branch = color_branch
        self.pass_pos_to_app_head = pass_pos_to_app_head
        out_dim += n_hyper
        if sdf_corrective:
            out_dim += 1
        self.n_hyper = n_hyper
        self.sdf_corrective = sdf_corrective
        print(f'Creating DeepSDF with input dim f{d_in}, hidden_dim f{hidden_dim} and output_dim {out_dim}')

        dims = [hidden_dim] * nlayers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [nlayers//2]
        self.num_freq_bands = num_freq_bands
        self.return_last_feats = return_last_feats
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))

        for layer in range(0, self.num_layers - 1):

            #if layer + 1 in self.skip_in:
            #    out_dim = dims[layer + 1] - d_in
            #else:
            if layer in self.skip_in:
                in_dim = dims[layer] + d_in
            else:
                in_dim = dims[layer]
            out_dim = dims[layer + 1]

            lin = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init:

                if layer == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                #else:
                #    torch.nn.init.constant_(lin.bias, 0.0)

                #    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #else:
            #    stdv = 1. / math.sqrt(lin.weight.size(1))
            #    #stdv = stdv / 5
            #    print('Attention: using lower std to init Linear layer!!')
            #    lin.weight.data.uniform_(-stdv, stdv)
            #    if lin.bias is not None:
            #        lin.bias.data.uniform_(-stdv, stdv)
            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

        if self.color_branch:
            self.return_last_feats = True

            d_in_color = hidden_dim + lat_dim_app
            if self.pass_pos_to_app_head:
                d_in_color += d_in_spatial
            self.color_mlp = nn.Sequential(
                nn.Linear(d_in_color, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )

    def forward(self, in_dict, anchors=None):

        xyz = in_dict['queries_can']
        lat_rep = in_dict['cond']['geo']
        lat_rep = lat_rep[:, :1, :]
        if lat_rep.shape[1] == 1 and xyz.shape[1] > 1:
            lat_rep = lat_rep.repeat(1, xyz.shape[1], 1)
        if self.color_branch:
            lat_rep_color = in_dict['cond']['app']
            lat_rep_color = lat_rep_color[:, :1, :]
            if lat_rep_color.shape[1] == 1 and xyz.shape[1] > 1:
                lat_rep_color = lat_rep_color.repeat(1, xyz.shape[1], 1)

        if len(xyz.shape) < 3:
            xyz = xyz.unsqueeze(0)

        batch_size, num_points, dim_in = xyz.shape

        if self.num_freq_bands is not None:
            pos_embeds = [xyz]
            for freq in self.freq_bands:
                pos_embeds.append(torch.sin(xyz* freq))
                pos_embeds.append(torch.cos(xyz * freq))

            pos_embed = torch.cat(pos_embeds, dim=-1)
            inp = torch.cat([pos_embed, lat_rep], dim=-1)
        else:
            inp = torch.cat([xyz, lat_rep], dim=-1)
        x = inp
        last_feats = None

        for layer in range(0, self.num_layers - 1):
            #print(x.shape)
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, inp], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)
            if self.return_last_feats and layer == self.num_layers - 3:
                last_feats = x

        if self.color_branch:
            if self.pass_pos_to_app_head:
                if self.num_freq_bands is not None:
                    color_cond = torch.cat([pos_embed, last_feats, lat_rep_color], dim=-1)
                else:
                    color_cond = torch.cat([xyz, last_feats, lat_rep_color], dim=-1)
            else:
                color_cond = torch.cat([last_feats, lat_rep_color], dim=-1)

            color_preds = self.color_mlp(color_cond)

            return {'sdf': x, 'anchors': None, 'color': color_preds}

        return {'sdf': x, 'anchors': None}


class EnsembleDeepSDF(torch.nn.Module):
    '''
    Implements a single layer of the local MLP ensemble used in NPHMs.
    '''
    def __init__(self,  ensemble_size: int,
                        n_symm: int,
                        lat_dim: int,
                        hidden_dim: int,
                        n_layers: int=8,
                        beta: int=100,
                        out_dim: int=1,
                        num_freq_bands: Optional[int]=None,
                        input_dim: int=3,
                        return_color_feats: bool=False):
        super().__init__()

        if num_freq_bands is None:
            d_in = input_dim + lat_dim
        else:
            d_in = input_dim * (2 * num_freq_bands + 1) + lat_dim
        self.ensemble_size = ensemble_size
        self.n_symm = n_symm
        self.lat_dim = lat_dim
        self.input_dim = input_dim
        self.return_color_feats = return_color_feats

        dims = [hidden_dim] * n_layers
        dims = [d_in] + dims + [out_dim]

        self.num_layers = len(dims)
        self.skip_in = [n_layers // 2]
        self.num_freq_bands = num_freq_bands
        self.num_types = self.ensemble_size - self.n_symm + 1
        #self.return_last_feats = return_last_feats
        if num_freq_bands is not None:
            fun = lambda x: 2 ** x
            self.freq_bands = fun(torch.arange(num_freq_bands))

        self.layers = torch.nn.ModuleList()

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                # out_dim = dims[layer + 1] #- d_in
                # in_dim = dims[layer] #+ d_in
                out_dim = dims[layer + 1] - d_in
                in_dim = dims[layer]
            else:
                out_dim = dims[layer + 1]
                in_dim = dims[layer]
            assert torch_geometric.typing.WITH_PYG_LIB
            lin = HeteroLinear(in_channels=in_dim,
                               out_channels=out_dim,
                               num_types=self.num_types,
                               is_sorted=True,
                               #TODO bias_initializer='uniform'
                               )
            self.layers.append(lin)

        #if self.return_color_feats:
        #    self.color_feats_extractor = HeteroLinear(in_channels=hidden_dim,
        #                                             out_channels=self.return_color_feats,
        #                                              num_types=self.num_types,
        #                                              is_sorted=True,
        #                                              bias_initializer='uniform')

        if beta > 0:
            self.activation = torch.nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = torch.nn.ReLU()

        self.reset_parameters()


    def forward(self, x: Tensor, type_vec: Tensor) -> (Tensor, Tensor):
        inp = x
        color_feats = None
        #print('layer', len(self.layers))
        for i, layer in enumerate(self.layers):
            #print('i', i)

            if i in self.skip_in:
                x = torch.cat([x, inp], -1) / 1.414

            x = layer(x, type_vec)


            if self.return_color_feats > 0 and i == len(self.layers) - 2:
                #print('running color extractor')
                color_feats = x #self.color_feats_extractor(x, type_vec)
                #print(color_feats.shape)
            if i < self.num_layers - 2:
                x = self.activation(x)


        return x, color_feats



    def reset_parameters(self):
        for lin in self.layers:
            lin.reset_parameters()


class NPHM(MessagePassing):
    '''
        Implements the SDF (and optionally texture field) in canonical space using an ensemble of local MLPs.
        In this class the modelling of a texture field (color_branch==True) is deprecated.
    '''
    def __init__(self,
                 lat_dim_glob: int,
                 lat_dim_loc: int,

                 n_symm: int,
                 n_anchors: int,
                 anchors: torch.Tensor,

                 hidden_dim: int,
                 n_layers: int,
                 n_layers_color: int = 4,
                 d_pos: int = 3,
                 d_out : int = 1,
                 pos_mlp_dim: int = 128,
                 num_neighbors: int = 8,
                 color_branch: bool = False,
                 num_freq_bands: Optional[int] = None,
                 color_communication : bool = True,
                 num_freq_bands_comm: Optional[int] = None,
                 disable_hetero_linear: bool = False,
                 is_monolith : bool = False,
                 const_anchors : bool = False,
                 color_pass_pos : bool = False,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.anchors = anchors.squeeze(0)

        self.color_branch = color_branch
        self.color_communication = color_communication
        self.color_pass_pos = color_pass_pos


        self.lat_dim_glob = lat_dim_glob
        self.lat_dim_loc = lat_dim_loc
        self.lat_dim_loc_geo = lat_dim_loc # alias

        if not self.color_branch:
            self.lat_dim_loc_modality = lat_dim_loc
        else:
            self.lat_dim_loc_modality = lat_dim_loc // 2
            #TODO
            self.lat_dim_geo = self.lat_dim_loc_modality
            self.lat_dim_app = self.lat_dim_loc_modality

        self.n_symm = n_symm
        self.num_symm_pairs = self.n_symm #alias
        self.num_kps = n_anchors #alias
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_layers_color = n_layers_color
        self.n_anchors = n_anchors
        self.pos_mlp_dim = pos_mlp_dim
        self.num_neighbors = num_neighbors
        self.lat_dim = lat_dim_glob + (n_anchors+1) * lat_dim_loc
        self.num_freq_bands = num_freq_bands if num_freq_bands is not None else 0
        self.num_freq_bands_comm = num_freq_bands_comm if num_freq_bands_comm is not None else 0
        self.d_out = d_out
        self.is_mololith = is_monolith
        self.const_anchors = const_anchors

        self.num_types = self.num_kps - self.n_symm + 1

        if self.color_communication:
            d_in_geo = self.lat_dim_loc_modality+self.lat_dim_glob
        else:
            #d_in_geo = self.lat_dim_loc_modality+self.lat_dim_glob
            d_in_geo = 2*self.lat_dim_loc_modality+self.lat_dim_glob
        if self.is_mololith:
            d_in_geo *= 2

        self.deepSDFensemble = EnsembleDeepSDF(ensemble_size=self.n_anchors,
                                               n_symm=self.n_symm,
                                               lat_dim=d_in_geo,
                                               hidden_dim=self.hidden_dim,
                                               n_layers=self.n_layers,
                                               return_color_feats=self.color_branch,
                                               input_dim=d_pos,
                                               out_dim=self.d_out,
                                               ).float()
        # deprecated
        if self.color_branch:
            if self.num_freq_bands > 0:
                fun = lambda x: 2 ** x
                self.freq_bands = fun(torch.arange(num_freq_bands))
            else:
                self.freq_bands = []

            if self.num_freq_bands_comm > 0:
                fun = lambda x: 2 ** x
                self.freq_bands_comm = fun(torch.arange(num_freq_bands_comm))
            else:
                self.freq_bands_comm = []

            if self.color_communication:
                #dim_in_color_mlp = hidden_dim + self.lat_dim_loc_modality + 3 + 3 * 2 * self.num_freq_bands
                communication_dim = 16
                self.communcation_bottleneck1 = HeteroLinear(num_types=self.num_types,
                                                            in_channels=hidden_dim,
                                                            out_channels=communication_dim,
                                                            is_sorted=True,
                                                            bias_initializer='uniform').float()
                self.activation_bottleneck = torch.nn.ReLU()
                self.communcation_bottleneck2 = HeteroLinear(num_types=self.num_types,
                                                            in_channels=communication_dim,
                                                            out_channels=communication_dim,
                                                            is_sorted=True,
                                                            bias_initializer='uniform').float()
                communication_dim += communication_dim * 2 * self.num_freq_bands_comm
                dim_in_color_mlp = communication_dim + self.lat_dim_loc_modality
                if self.color_pass_pos:
                    dim_in_color_mlp += 3 + 3 * 2 * self.num_freq_bands
            else:
                # no comm. separated dim_in_color_mlp = self.lat_dim_loc_modality
                dim_in_color_mlp = 2*self.lat_dim_loc_modality + 3 + 3 * 2 * self.num_freq_bands


            color_layers = [HeteroLinear(num_types=self.num_types,
                                in_channels=dim_in_color_mlp,
                                out_channels=hidden_dim,
                                is_sorted=True,
                                bias_initializer='uniform'
                                ).float()]
            for _ in range(self.n_layers_color-2):
                color_layers.append(HeteroLinear(num_types=self.num_types,
                                in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                is_sorted=True,
                                bias_initializer='uniform'
                             ).float())
            color_layers.append(HeteroLinear(num_types=self.num_types,
                                in_channels=hidden_dim,
                                out_channels=3,
                                is_sorted=True,
                                bias_initializer='uniform'
                             ).float())
            self.ensembled_color_mlp = torch.nn.ModuleList(color_layers)

            self.activation = torch.nn.ReLU()



        self.reset_parameters()
        self.is_symm = torch.ones([self.n_symm*2], dtype=torch.bool)
        self.is_symm[::2] = 0
        self.is_symm = torch.cat([self.is_symm, torch.zeros(self.n_anchors - 2*self.n_symm + 1, dtype=torch.bool, device=self.is_symm.device)])
        self.is_symm = self.is_symm.unsqueeze(-1).unsqueeze(0) #.cuda()
        self.anchor_idx = torch.repeat_interleave(torch.arange(self.n_symm), 2)
        self.anchor_idx = torch.cat([self.anchor_idx, torch.arange(self.n_symm, self.n_anchors - self.n_symm + 1)])
        self.anchor_idx = self.anchor_idx.unsqueeze(-1).unsqueeze(0) #.cuda()

        if not self.const_anchors:
            self.mlp_pos = torch.nn.Sequential(
                torch.nn.Linear(self.lat_dim_glob, self.pos_mlp_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.pos_mlp_dim, self.pos_mlp_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.pos_mlp_dim, self.n_anchors * 3)
            )

        propagate_type = {'feats': Tuple[Tensor, Tensor], 'color_feats': Tuple[Tensor, Tensor],
                          'glob_feats': Tuple[Tensor, Tensor], 'anchor_idx': Tensor, 'is_symm': Tensor, 'pos': Tuple[Tensor, Tensor],
                          'debug_region': bool}

    def reset_parameters(self):
        super().reset_parameters()

        for lin in self.deepSDFensemble.layers:
            lin.reset_parameters()

        if self.color_branch:
            for lin in self.ensembled_color_mlp:
                lin.reset_parameters()



    def forward(self, in_dict, debug_regio=False) -> (Tensor, Tensor):
        '''
        Executes the SDF (and texture field) in canonical space.
        Inclusion of a texture field is not supported anymore in here.

        :param in_dict: holdes the queries in canonical space, as well as, the latent codes for geometry and appearance.
        :param debug_regio: Should always be False. If True, the kNN edges and weights are cached for later visualization.
        :return: Reutruns a dictionary of SDF predictions. Also, the predicted anchor positions are included.
                If a texture field is represented color values are included.
        '''

        query_pos = in_dict['queries_can'] # Either B x dim_can or B x num_queries x dim_can
        lat_rep = in_dict['cond']['geo'] # Either B x dim_can or B x num_queries x lat_dim_geo
        # Not supported; in the monolithic case, latent expression codes are needed here, since there is no deformation field.
        if self.is_mololith:
            lat_rep_expr = in_dict['cond']['exp'] # Either B x dim_can or B x num_queries x lat_dim_exp

        device = query_pos.device

        if len(query_pos.shape) < 3:
            query_pos = query_pos.unsqueeze(0)

        batch_size, num_points, dim_in = query_pos.shape

        # extract global geometry codes
        glob_feats = lat_rep[:, 0, :self.lat_dim_glob]


        # not supported properly!
        if self.color_branch:

            if self.color_communication:
                anchor_feats = lat_rep[:, 0, self.lat_dim_glob::2].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
                color_feats = lat_rep[:, 0, self.lat_dim_glob+1::2].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
            else:
                anchor_feats = lat_rep[:, 0, self.lat_dim_glob:].reshape(batch_size * (self.n_anchors + 1),
                                                                           2*self.lat_dim_loc_modality)
                color_feats = lat_rep[:, 0, self.lat_dim_glob:].reshape(batch_size * (self.n_anchors + 1),
                                                                              2*self.lat_dim_loc_modality)
        else:
            # extracted local codes
            anchor_feats = lat_rep[:, 0, self.lat_dim_glob:].reshape(batch_size*(self.n_anchors+1), self.lat_dim_loc_modality)
            color_feats = None

        # not supported
        if self.is_mololith:
            assert not self.color_branch

            glob_feats_expr = lat_rep_expr[:, 0, :self.lat_dim_glob]
            anchor_feats_expr = lat_rep_expr[:, 0, self.lat_dim_glob:].reshape(batch_size * (self.n_anchors + 1),
                                                                     self.lat_dim_loc_modality)


        if not debug_regio:
            anchor_pos = self.get_anchors(lat_rep)
        else:
            anchor_pos = self.anchors

        # not supported
        if self.is_mololith:
            glob_feats = torch.cat([glob_feats, glob_feats_expr], dim=-1)
            glob_feats = (glob_feats.unsqueeze(1).repeat(1, self.n_anchors+1, 1).view(-1, self.lat_dim_glob*2), None)
            anchor_feats = torch.cat([anchor_feats, anchor_feats_expr], dim=-1)
        else:
            glob_feats = (glob_feats.unsqueeze(1).repeat(1, self.n_anchors+1, 1).view(-1, self.lat_dim_glob), None)
        anchor_feats = (anchor_feats,  None)
        color_feats = (color_feats, None)

        # prepare computation of  bi-partite knn graph

        # add batch dimension to anchor position if it is missing
        if query_pos.shape[0] > 1 and anchor_pos.shape[0] == 1:
            repeat_args = [1 for _ in range(len(anchor_pos.shape))]
            repeat_args[0] = query_pos.shape[0]
            anchor_pos = anchor_pos.repeat(*repeat_args)

        # compute k-nearest-neighbors from queries to anchors
        knn_results = knn_points(query_pos[..., :3], anchor_pos, K=self.num_neighbors)


        # the "last" MLP in the ensemble, which is missing a cooresponding anchor, is always included
        knn_idx = torch.cat([knn_results.idx, (self.n_anchors) * torch.ones_like(knn_results.idx[:, :, :1])], dim=-1)

        # build bipartit knn graph

        anchor_idx_correction = torch.arange(batch_size, device=query_pos.device).unsqueeze(1).unsqueeze(1).to(device) * (self.n_anchors + 1)
        point_idx_correction = torch.arange(batch_size, device=query_pos.device).unsqueeze(1).to(device) * (num_points)
        knn_edges = torch.stack([(knn_idx + anchor_idx_correction).view(batch_size, -1),
                                 torch.repeat_interleave(torch.arange(num_points, device=query_pos.device),
                                                         self.num_neighbors+1, dim=0).unsqueeze(0).repeat(batch_size, 1) + point_idx_correction], dim=1).to(device)

        # merge batch and point-sample dimensions
        edge_index = knn_edges.permute(1, 0, 2).reshape(2, -1)

        if debug_regio:
            self.latest_edges = knn_edges

        # prepare message passing from anchors to queries, i.e. the queries is where information is aggregated
        pos = (torch.cat([anchor_pos, torch.zeros_like(anchor_pos[:, :1, :])], dim=1), query_pos)

        # need to make tensors 2-dimensional for message-passing of PytorchGeometric
        pos = (pos[0].view(-1, 3), pos[1].view(-1, dim_in))

        anchor_idx = (self.anchor_idx.repeat(batch_size, 1, 1).view(-1, 1).to(device), None)
        is_symm = (self.is_symm.repeat(batch_size, 1, 1).view(-1, 1).to(device), None)

        # perform message passing
        out = self.propagate(edge_index,
                             feats=anchor_feats,
                             color_feats=color_feats,
                             glob_feats=glob_feats,
                             anchor_idx=anchor_idx,
                             is_symm=is_symm,
                             pos=pos,
                             debug_region=debug_regio,
                             size=None)

        # bring back batch and query-points dimensions
        out = out.view(batch_size, num_points, -1)

        # not supported; split SDF and color predictions
        if self.color_branch:
            out_color = out[..., self.d_out:]
            out = out[..., :self.d_out]

            return {'sdf': out, 'anchors': anchor_pos, 'color': out_color}
        else:
            return {'sdf': out, 'anchors': anchor_pos}



    def message(self,
                feats_j: Tensor,
                color_feats_j,
                glob_feats_j: Tensor,
                anchor_idx_j: Tensor,
                is_symm_j: Tensor,
                pos_i: Tensor,
                pos_j: Tensor,
                #global_pred_i: Tensor,
                index: Tensor,
                ptr: OptTensor,
                debug_region: bool=False) -> (Tensor, Tensor):
        '''
        Implementation of the message passing.
        Inputs have at most two dimensions.
        The first dimension is over all edges.
        "anchor_idx_j" holds information over the type of edge, i.e. the index of the involved anchor,
            which decides which MLP will be used to perform the computation. Remember, that each anchor has an
            individual set of MLP weights.

        "index" holds information of the local neighborhoods, which is used to perform the final aggregation.

        :param feats_j:
        :param color_feats_j:
        :param glob_feats_j:
        :param anchor_idx_j:
        :param is_symm_j:
        :param pos_i:
        :param pos_j:
        :param index:
        :param ptr:
        :param debug_region:
        :return:
        '''

        # move all queries in the local coordinate system of its associated anchor point
        delta = pos_i[..., :3] - pos_j

        # mirror symmetric points
        is_symmetric_idx = is_symm_j
        delta[is_symmetric_idx.squeeze(), 0] *= -1

        # if hyper dimensions are present, they are append to the input
        if pos_i.shape[-1] > 3:
            net_in = torch.cat([glob_feats_j, feats_j, delta, pos_i[..., 3:]], dim=-1)
        else:
            net_in = torch.cat([glob_feats_j, feats_j, delta], dim=-1)


        # sort anchor index here once due to nature of hetero linear implementation
        anchor_idx_j = anchor_idx_j.squeeze()
        anchor_idx_j_sorted, perm = index_sort(anchor_idx_j, self.deepSDFensemble.num_types)
        net_in = net_in[perm]
        # evaluate individual MLPs, each making an SDF prediction, which are blended later
        out_sorted, color_communication = self.deepSDFensemble(net_in, anchor_idx_j_sorted)

        # not supported
        if self.color_branch:
            net_in_color =  color_feats_j[perm]
            _delta = delta[perm]
            if self.num_freq_bands is not None:
                delta_embeds = [_delta]
                for freq in self.freq_bands:
                    delta_embeds.append(torch.sin(_delta * freq))
                    delta_embeds.append(torch.cos(_delta * freq))

                delta_embeds = torch.cat(delta_embeds, dim=-1)
            else:
                delta_embeds = _delta
            if self.color_communication:
                #net_in_color = torch.cat([net_in_color, color_communication, delta_embeds], dim=-1)
                color_communication = self.communcation_bottleneck1(color_communication, anchor_idx_j_sorted)
                color_communication = self.activation_bottleneck(color_communication)
                color_communication = self.communcation_bottleneck2(color_communication, anchor_idx_j_sorted)
                if self.num_freq_bands_comm > 0:
                    color_communication_embes = [color_communication]
                    for freq in self.freq_bands_comm:
                        color_communication_embes.append(torch.sin(color_communication * freq))
                        color_communication_embes.append(torch.cos(color_communication * freq))
                    color_communication = torch.cat(color_communication_embes, dim=-1)

                if self.color_pass_pos:
                    net_in_color = torch.cat([net_in_color, color_communication, delta_embeds], dim=-1)
                else:
                    net_in_color = torch.cat([net_in_color, color_communication], dim=-1)
            else:
                net_in_color = torch.cat([net_in_color, delta_embeds], dim=-1)

            x = net_in_color
            for lin_idx, lin in enumerate(self.ensembled_color_mlp):
                x = lin(x, anchor_idx_j_sorted)
                if lin_idx < len(self.ensembled_color_mlp) - 1:
                    x = self.activation(x)
            out_color = x

        # undo sorting
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        out = out_sorted[inv, :]
        if self.color_branch:
            out_color = out_color[inv, :]

        # compute distance of queries to associated anchors
        _distances = torch.norm(delta, dim=-1)
        # the "last" MLP doesn't have an anchor, hence a constant large distance is assigned
        # The "last" MLP is used to model stuff far away from all anchors
        distances = torch.where(anchor_idx_j  == (self.n_anchors-self.n_symm), 0.75, _distances)

        # blend all SDF predictions for a query point
        alpha = softmax_mod_varySize2(distances, index, ptr, num_neighbors=self.num_neighbors)

        if debug_region:
            self.latest_wights = alpha

        if self.color_branch:
            return alpha.unsqueeze(-1) * torch.cat([out,  out_color], dim=-1)
        else:
            return alpha.unsqueeze(-1) * out

    def get_anchors(self, lat_rep_geo : Tensor):
        '''
        Predict the acnhor positions based on the global part of the geometry latent code.

        :param lat_rep_geo:
        :return:
        '''
        if not self.const_anchors:
            glob_feats = lat_rep_geo[:, 0, :self.lat_dim_glob]
            offsets = self.mlp_pos(glob_feats).view(-1, self.n_anchors, 3)
            return self.anchors.to(offsets.device) + offsets
        else:
            return self.anchors.to(lat_rep_geo.device)


    def get_symm_reg(self, cond, cond_type : str):
        '''
        Compute the difference between symmetric local latent vectors.

        :param cond:
        :param cond_type:
        :return:
        '''
        if cond_type == 'geo':
            shape_dim_glob = self.lat_dim_glob
            shape_dim_loc = self.lat_dim_loc_geo
        elif cond_type == 'app':
            shape_dim_glob = self.lat_dim_glob
            shape_dim_loc = self.lat_dim_loc_app
        n_symm = self.num_symm_pairs
        loc_lats_symm = cond[:, shape_dim_glob:shape_dim_glob + 2 * n_symm * shape_dim_loc].view(
            cond.shape[0], n_symm * 2, shape_dim_loc)
        loc_lats_middle = cond[:, shape_dim_glob + 2 * n_symm * shape_dim_loc:-shape_dim_loc].view(
            cond.shape[0], self.num_kps - n_symm * 2, shape_dim_loc)

        symm_dist = torch.norm(loc_lats_symm[:, ::2, :] - loc_lats_symm[:, 1::2, :], dim=-1).mean()
        if loc_lats_middle.shape[1] % 2 == 0:
            middle_dist = torch.norm(loc_lats_middle[:, ::2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
        else:
            middle_dist = torch.norm(loc_lats_middle[:, :-1:2, :] - loc_lats_middle[:, 1::2, :], dim=-1).mean()
        return symm_dist, middle_dist


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})') # TODO


def get_id_model(cfg_id,
                 device : torch._C.device = 0,
                 ):
    '''
    Instantiate NPHM model (used as SDF in canonical space).

    :param cfg_id:
    :param device:
    :return:
    '''

    anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH.format(cfg_id['nloc']))).float().unsqueeze(0).unsqueeze(0).to(device)

    return NPHM(lat_dim_glob=cfg_id['lat_dim_glob'],
                lat_dim_loc=cfg_id['lat_dim_loc_geo'],
                n_symm=cfg_id['nsymm_pairs'],
                n_anchors=cfg_id['nloc'],
                anchors=anchors,
                hidden_dim=cfg_id['gnn']['hidden_dim_geo'],
                n_layers=cfg_id['gnn']['nlayers_geo'],
                n_layers_color=cfg_id['gnn']['nlayers_app'],
                num_neighbors=cfg_id['nneigh'],
                d_pos=3 + cfg_id['n_hyper'],
                ), anchors
