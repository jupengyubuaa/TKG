# -*- coding=utf-8-*-
# @Time: 2022/12/23 14:29
# @Author: 鞠鹏羽
# @File: His.py
# @Software: PyCharm

# 首先确定历史窗口大小h
# 从t-h  到 t-1 提取O 保留顺序 可参考CyGNET  其记录index的为1 我们则是直接记录index。所以应该是一个index序列
# 从构建的图中获得
import networkx as nx
from collections import defaultdict
import numpy as np
import math
from typing import Tuple
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder,TransformerEncoderLayer

h=24   #(历史窗口大小为24)
transformer_input_max=25 #即最大的候选实体序列长度，不足则需要补全，也有可能超了？那就要进行重排序，shape为[25,d],补全部分自然emdedding数值为0


print(np.power(0.5,2))
##############目前为列表存储且保留顺序，在后续进入transformer进行编码时,应参考进行padding或参考变长的编码方式
import numpy as np
import torch
import torch.nn as nn

class Episode(nn.Module):
    def __init__(self, env, agent, config):
        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']
        self.max_action_num = config['max_action_num']
        self.max_candidate_num=config['max_candidate_num']
        self.ent_dim=config['ent_dim']
        self.ePAD = config['num_ent']  # Padding entity

    def forward(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]
        """
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)
        history_candidate_space=self.env.get_history_space(
        query_entities,
        query_relations,
        query_timestamps,
        self.max_candidate_num)

        #Pad Mask for history_candidate_space
        candidate_mask=torch.ones_like(history_candidate_space[:,:,0])*self.ePAD #[batch_size,candidate_num]
        history_mask=torch.eq(candidate_mask,history_candidate_space[:,0,0].unsqueeze(1))
        candidate_mask=torch.eq(history_candidate_space[:,:,0],candidate_mask)   #[batch_size,candidate_num]            标志着candidate是否为padding过的
        candidate_mask=torch.logical_xor(candidate_mask,history_mask)

        candidate_num=history_candidate_space.size(1)
        candidate_delta_time=query_timestamps.unsqueeze(-1).repeat(1,candidate_num)-history_candidate_space[:,:,1]
        candidate_embds=self.agent.ent_embs(history_candidate_space[:,:,0],candidate_delta_time)

        current_entities = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        all_loss = []
        all_logits = []
        all_actions_idx = []
        all_path_score=[]
        Hq = self.agent.TransformerEncoder(candidate_embds, candidate_mask)

        self.agent.policy_step.set_hiddenx(query_relations.shape[0])
        ###################################设置lstm隐层参数#################################
        for t in range(self.path_length):
            if t == 0:
                first_step = True

            else:
                first_step = False

            action_space = self.env.next_actions(                                                            #action_space来自于cuda
                current_entities,
                current_timestamps,
                query_timestamps,
                self.max_action_num,
                first_step
            )
            #############################################################################
            loss, logits, action_id,path_score = self.agent(
                prev_relations,
                current_entities,
                current_timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
                Hq,
                history_mask
            )
            ################################################################################
            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_path_score=torch.gather(path_score,dim=1,index=action_id)

            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)
            all_path_score.append(chosen_path_score)

            current_entities = chosen_entity
            current_timestamps = chosen_entity_timestamps
            prev_relations = chosen_relation
        return all_loss, all_logits, all_actions_idx, current_entities, current_timestamps,all_path_score


    def beam_search(self, query_entities, query_timestamps, query_relations):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            current_entities: [batch_size, test_rollouts_num]
            beam_prob: [batch_size, test_rollouts_num]
        """
        batch_size = query_entities.shape[0]
        query_entities_embeds = self.agent.ent_embs(query_entities, torch.zeros_like(query_timestamps))
        query_relations_embeds = self.agent.rel_embs(query_relations)
        history_candidate_space=self.env.get_history_space(
        query_entities,
        query_relations,
        query_timestamps,
        self.max_candidate_num)

        #Pad Mask for history_candidate_space
        candidate_mask=torch.ones_like(history_candidate_space[:,:,0])*self.ePAD #[batch_size,candidate_num]
        history_mask=torch.eq(candidate_mask,history_candidate_space[:,0,0].unsqueeze(1))
        candidate_mask=torch.eq(history_candidate_space[:,:,0],candidate_mask)   #[batch_size,candidate_num]            标志着candidate是否为padding过的
        candidate_mask=torch.logical_xor(candidate_mask,history_mask)

        candidate_num=history_candidate_space.size(1)
        candidate_delta_time=query_timestamps.unsqueeze(-1).repeat(1,candidate_num)-history_candidate_space[:,:,1]
        candidate_embds=self.agent.ent_embs(history_candidate_space[:,:,0],candidate_delta_time)
        Hq = self.agent.TransformerEncoder(candidate_embds, candidate_mask)


        self.agent.policy_step.set_hiddenx(batch_size)

        # In the first step, if rollouts_num is greater than the maximum number of actions, select all actions
        current_entities = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP
        action_space = self.env.next_actions(current_entities, current_timestamps,
                                             query_timestamps, self.max_action_num, True)
        loss, logits, action_id, path_score = self.agent(
            prev_relations,
            current_entities,
            current_timestamps,
            query_relations_embeds,
            query_entities_embeds,
            query_timestamps,
            action_space,
            Hq,
            history_mask
        )  # logits.shape: [batch_size, max_action_num]
        #################################################beam_size 要与最大动作数相比较######################################
        action_space_size = action_space.shape[1]
        if self.config['beam_size'] > action_space_size:
            beam_size = action_space_size
        else:
            beam_size = self.config['beam_size']
        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size, dim=1)  # beam_log_prob.shape [batch_size, beam_size]

        beam_log_prob = beam_log_prob.reshape(-1)  # [batch_size * beam_size]

        current_entities = torch.gather(action_space[:, :, 1], dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        current_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        prev_relations = torch.gather(action_space[:, :, 0], dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        self.agent.policy_step.hx = self.agent.policy_step.hx.repeat(1, 1, beam_size).reshape(
            [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
        self.agent.policy_step.cx = self.agent.policy_step.cx.repeat(1, 1, beam_size).reshape(
            [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]

        beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0)  # [batch_size * beam_size, max_action_num]
        for t in range(1, self.path_length):
            query_timestamps_roll = query_timestamps.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, 1, beam_size)
            query_entities_embeds_roll = query_entities_embeds_roll.reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, ent_dim]
            query_relations_embeds_roll = query_relations_embeds.repeat(1, 1, beam_size)
            query_relations_embeds_roll = query_relations_embeds_roll.reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, rel_dim]
            ########################################beam_size 其实内容重复的行#########################################################################
            action_space = self.env.next_actions(current_entities, current_timestamps,
                                                 query_timestamps_roll, self.max_action_num)
            #################################################################################################################################
            loss, logits, action_id = self.agent(
                prev_relations,
                current_entities,
                current_timestamps,
                query_relations_embeds_roll,
                query_entities_embeds_roll,
                query_timestamps_roll,
                action_space
            )  # logits.shape [bs * rollouts_num, max_action_num]

            hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)

            beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1,
                                                                              0)  # [batch_size * beam_size, max_action_num]
            beam_tmp += logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)  # [batch_size, beam_size * max_actions_num]

            if action_space_size * beam_size >= self.config['beam_size']:
                beam_size = self.config['beam_size']
            else:
                beam_size = action_space_size * beam_size

            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, beam_size, dim=1)  # [batch_size, beam_size]
            offset = top_k_action_id // action_space_size  # [batch_size, beam_size]

            offset = offset.unsqueeze(-1).repeat(1, 1, self.config[
                'state_dim'])  # [batch_size, beam_size]  不应该是[batch_size,beam_size,state_dim]???
            self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset)
            self.agent.policy_step.hx = self.agent.policy_step.hx.reshape([batch_size * beam_size, -1])
            self.agent.policy_step.cx = torch.gather(cx_tmp, dim=1, index=offset)
            self.agent.policy_step.cx = self.agent.policy_step.cx.reshape([batch_size * beam_size, -1])

            current_entities = torch.gather(action_space[:, :, 1].reshape(batch_size, -1), dim=1,
                                            index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space[:, :, 2].reshape(batch_size, -1), dim=1,
                                              index=top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space[:, :, 0].reshape(batch_size, -1), dim=1,
                                          index=top_k_action_id).reshape(-1)

            beam_log_prob = top_k_log_prob.reshape(-1)  # [batch_size * beam_size]

        return action_space[:, :, 1].reshape(batch_size, -1), beam_tmp