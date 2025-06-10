import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import permutations, product
from typing import Tuple, List, Dict
import json
from tqdm.auto import tqdm

import torch

from argoverse.map_representation.map_api import ArgoverseMap

import sys
sys.path.append('ISE')
sys.path.append('ISE/datasets')

from models.ise import ISE

from datasets.argoverse_v1_dataset import process_argoverse, get_lane_features, ArgoverseV1Dataset
from utils import TemporalData

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.evaluation.competition_util import generate_forecasting_h5


checkpoint_path = 'path-to-ckpt'
model = ISE.load_from_checkpoint(checkpoint_path=checkpoint_path, parallel=False, map_location=torch.device('cpu'))

split = 'test'
dataset = ArgoverseV1Dataset(root='/root', split=split, local_radius=model.hparams.local_radius)
output_all_k6 = {}
probs_all = {}
for i, inp in enumerate(tqdm(dataset)):
    x = inp.x.numpy()
    # y = inp.y.numpy()
    seq_id = inp.seq_id
    positions = inp.positions.numpy()

    # the location of the ego vehicle at TIMESTAMP=19
    origin = inp.origin.numpy().squeeze()
    
    # the index of the focal agent
    agent_index = inp.agent_index
    
    # ego_heading at TIMESTAMP=19 
    ego_heading = inp.theta.numpy()
    
    ro_angle = inp.rotate_angles[agent_index].numpy()
    
    # Global rotation to align with ego vehicle
    rotate_mat = np.array([
        [np.cos(ego_heading), -np.sin(ego_heading)],
        [np.sin(ego_heading), np.cos(ego_heading)]
    ])
    
    R =  np.array([
                    [np.cos(ro_angle), -np.sin(ro_angle)],
                    [np.sin(ro_angle), np.cos(ro_angle)]
                ])
   

    # we recover the agent trajectory from the inputs, just as a sanity check
    offset = positions[agent_index, 19, :]
    hist = (np.cumsum(-x[agent_index, 20::-1, :], axis=0)[::-1, :] + offset) @ rotate_mat.T + origin
    # fut =  (y[agent_index, :, :] + offset) @ rotate_mat.T + origin
    
    res, res_pi = model(inp)
    agt_res = res[:, agent_index, :, :].detach().cpu().numpy() # [6, num_agents, 30, 2]

    probs = torch.softmax(res_pi[agent_index], dim=0)
    
    agt_res_origin = (agt_res[:, :, :2] @ R.T + offset) @ rotate_mat.T + origin
    
    probs_all[seq_id] = probs.detach().cpu().numpy()
    output_all_k6[seq_id] = agt_res_origin

output_path = 'competition_files/'

generate_forecasting_h5(output_all_k6, output_path, probabilities= probs_all, filename = 'test1') #this might take awhile