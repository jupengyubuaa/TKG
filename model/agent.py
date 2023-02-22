import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import TransformerEncoder,TransformerEncoderLayer

class Encoder(nn.Module):
    # def __init__(self,d_model,nhead,d_hid,n_layers,dropout):
    def __init__(self,config):
        super(Encoder, self).__init__()
        self.d_model=config['ent_dim']
        self.dropout=config['dropout']
        self.nhead=config['nhead']
        self.d_hid=config['d_hid']
        self.n_layers=config['n_layers']
        self.his_length=config['max_candidate_num']
        self.model_type='Transformer'
        self.pos_encoder=PositionalEncoding(self.d_model,self.dropout,self.his_length)
        encoder_layers=TransformerEncoderLayer(self.d_model,self.nhead,self.d_hid,self.dropout,batch_first=True)        #定义编码器每一层
        self.transformer_encoder=TransformerEncoder(encoder_layers,self.n_layers)                                       #完整编码器，多层

    # def forward(self,history_candidate_space,mask):
    def forward(self,history_candidate_space,mask):
        """
        Args:
            history_candidate_space:[batch_size,max_candidate_num,ent_dim]
        Return:
            Hq:[batch_size,state_dim]
        """
        self.mask=mask
        history_candidate=history_candidate_space*math.sqrt(self.d_model)                                          #通过平方重新分布
        history_candidate=self.pos_encoder(history_candidate)                                                      #位置编码
        output=self.transformer_encoder(src=history_candidate,src_key_padding_mask=self.mask)
        # output=self.transformer_encoder(src=history_candidate)
        output=output[:,0,:]
        return output
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len):                                                            #d_model注意要与ent_dim相同
        super(PositionalEncoding, self).__init__()
        self.dropout=nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe=torch.zeros(1,max_len,d_model)                                                                        #此处设计初衷为最大
        pe[0,:,0::2]=torch.sin(position*div_term)
        pe[0,:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    def forward(self,x):
        """
        Args:
            x:Tensor,shape[batch_size,max_candidate_num,embedding_dim]
        """
        x=x+self.pe                                                                                             #即此处为后padding,顺序具有巨大不同
        return self.dropout(x)
class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super(HistoryEncoder, self).__init__()
        self.config = config
        self.lstm_cell = torch.nn.LSTMCell(input_size=config['action_dim'],
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        self.mlp_l1= nn.Linear(config['mlp_input_dim'], config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['action_dim'], bias=True)
        ##############################################需要修改的模块，即mlp#################################
    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        return output

class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t):
        super(DynamicEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())             #得到dim_t 维度的数组
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())                                       #dim_t维的数组

########################################################################################################
    def forward(self, entities, dt):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)
        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]

        e = self.ent_embs(entities)
        return torch.cat((e, t), -1)
    ##################格式#################################################################################

class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)

    def forward(self, entities, timestamps=None):
        return self.ent_embs(entities)

class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.num_rel = config['num_rel'] * 2 + 2
        self.config = config
        self.TransformerEncoder=Encoder(config)
        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        # self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.NO_OP=config['num_rel']

        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation    reletion为2n+2的原因
        self.tPAD = 0  # Padding time   意义应该是相对时间差

        if self.config['entities_embeds_method'] == 'dynamic':
            self.ent_embs = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'])
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'])

        self.rel_embs = nn.Embedding(config['num_ent'], config['rel_dim'])

        self.policy_step = HistoryEncoder(config)
        self.policy_mlp = PolicyMLP(config)
        ###################################policy相关##########################################################
        # self.score_weighted_fc = nn.Linear(
        #     self.config['ent_dim'] * 2 + self.config['rel_dim'] * 2 + self.config['state_dim'],
        #     1, bias=True)
        self.score_weighted_fc = nn.Linear(self.config['rel_dim'] * 2 + self.config['state_dim'],1, bias=True)     #得分计算尝试
        self.his_score_weighted = nn.Linear(self.config['rel_dim'] + self.config['ent_dim'],1, bias=True)     #得分计算尝试

    def forward(self, prev_relation, current_entities, current_timestamps,
                query_relation, query_entity, query_timestamps, action_space,Hq,history_mask,t):
        """
        Args:
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation，[batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps: [batch_size]
            action_space: [batch_size, max_actions_num, 3] (relations, entities, timestamps)
        """
        # embeddings
        current_delta_time = query_timestamps - current_timestamps
        current_embds = self.ent_embs(current_entities, current_delta_time)  # [batch_size, ent_dim]
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # Pad Mask
        pad_mask = torch.ones_like(action_space[:, :, 0]) * self.rPAD  # [batch_size, action_number]
        pad_mask = torch.eq(action_space[:, :, 0], pad_mask)  # [batch_size, action_number]


        # History Encode
        NO_OP_mask = torch.eq(prev_relation, torch.ones_like(prev_relation) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]
        prev_action_embedding = torch.cat([prev_relation_embds, current_embds], dim=-1)  # [batch_size, rel_dim + ent_dim]
        lstm_output = self.policy_step(prev_action_embedding, NO_OP_mask)  # [batch_size, state_dim]

        # Neighbor/condidate_actions embeddings
        action_num = action_space.size(1)
        neighbors_delta_time = query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        neighbors_entities = self.ent_embs(action_space[:, :, 1], neighbors_delta_time)  # [batch_size, action_num, ent_dim]
        neighbors_relations = self.rel_embs(action_space[:, :, 0])  # [batch_size, action_num, rel_dim]

        # agent state representation
        agent_state = torch.cat([lstm_output, query_entity, query_relation,Hq], dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim+ent_dim]   得分计算二
        # agent_state = torch.cat([lstm_output, query_entity, query_relation], dim=-1)  # [batch_size, state_dim + ent_dim + rel_dim]
        output = self.policy_mlp(agent_state)  # [batch_size, 1, action_dim] action_dim == rel_dim + ent_dim
        ##############################################状态嵌入#######################################################
        # scoring
        entitis_output = output[:, :, self.config['rel_dim']:]
        relation_ouput = output[:, :, :self.config['rel_dim']]

        Hq = Hq.unsqueeze(1)
        entities_history_score=torch.sum(torch.mul(neighbors_entities, Hq), dim=2)  # [batch_size, 1,action_number]
        entities_history_score=entities_history_score.masked_fill(history_mask, 0)
        #
        query_relation_repeats=query_relation.unsqueeze(1).repeat(1,neighbors_relations.shape[1],1)
        relation_score = torch.sum(torch.mul(neighbors_relations, query_relation_repeats-lstm_output.unsqueeze(1)), dim=2)
        relation_score_weight=torch.sigmoid(relation_score)
        relation_score = torch.sum(torch.mul(neighbors_relations, relation_ouput), dim=2)
        entities_score = torch.sum(torch.mul(neighbors_entities, entitis_output), dim=2)  # [batch_size, action_number]

        # actions = torch.cat([neighbors_relations, neighbors_entities], dim=-1)  # [batch_size, action_number, action_dim]
        # agent_state_repeats = agent_state.unsqueeze(1).repeat(1, actions.shape[1], 1)
        # score_attention_input = torch.cat([actions, agent_state_repeats], dim=-1)
        # a = self.score_weighted_fc(score_attention_input)
        # a = torch.sigmoid(a).squeeze()  # [batch_size, action_number]
        # scores = (1 - a) * relation_score + a * entities_score

        lstm_output_repeats=lstm_output.unsqueeze(1).repeat(1,neighbors_relations.shape[1],1)
        score_attention_input = torch.cat([lstm_output_repeats, query_relation_repeats,neighbors_relations], dim=-1)  # [batch_size, action_number, action_dim]
        a = self.score_weighted_fc(score_attention_input)
        b = self.his_score_weighted(torch.cat([query_entity,query_relation],dim=1))
        a = torch.sigmoid(a).squeeze()  # [batch_size, action_number]
        b = torch.sigmoid(b)
        ###############################################计算得分可解释性是否解释得通##########################################

        # print(torch.mean(b))
        if t==2:
            scores = relation_score_weight * relation_score + (1-relation_score_weight) * entities_score + b * entities_history_score
        else:
            scores = relation_score_weight * relation_score + (1-relation_score_weight) * entities_score
        # scores = b * relation_score+(1-b) * entities_score
        # scores = relation_score_weight * relation_score + (1-relation_score_weight) * entities_score + b * entities_history_score
        # scores = relation_score_weight * relation_score + (1-relation_score_weight) * entities_score
        # print(torch.mean(relation_score_weight))
        # Padding mask
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]
        action_prob = torch.softmax(scores, dim=1)

        # 观察a与scores 的数量级差距，进行调整。

        action_id = torch.multinomial(action_prob, 1)  # Randomly select an action. [batch_size, 1]
        logits = torch.nn.functional.log_softmax(scores, dim=1)  # [batch_size, action_number]
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)
        return loss, logits, action_id

    def get_im_embedding(self, cooccurrence_entities):
        """Get the inductive mean representation of the co-occurrence relation.
        cooccurrence_entities: a list that contains the trained entities with the co-occurrence relation.
        return: torch.tensor, representation of the co-occurrence entities.
        """
        entities = self.ent_embs.ent_embs.weight.data[cooccurrence_entities]
        im = torch.mean(entities, dim=0)
        return im
        #######################################考虑的是实体更新，那我的舆情研究就派上了用场，要素##############################
    def update_entity_embedding(self, entity, ims, mu):
        """Update the entity representation with the co-occurrence relations in the last timestamp.
        entity: int, the entity that needs to be updated.
        ims: torch.tensor, [number of co-occurrence, -1], the IM representations of the co-occurrence relations
        mu: update ratio, the hyperparam.
        """
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * torch.mean(ims, dim=0)

    def entities_embedding_shift(self, entity, im, mu):
        """Prediction shift."""
        self.source_entity = self.ent_embs.ent_embs.weight.data[entity]
        self.ent_embs.ent_embs.weight.data[entity] = mu * self.source_entity + (1 - mu) * im

    def back_entities_embedding(self, entity):
        """Go back after shift ends."""
        self.ent_embs.ent_embs.weight.data[entity] = self.source_entity

