import networkx as nx
from collections import defaultdict
import numpy as np
import torch

class Env(object):
    def __init__(self, examples, config, state_action_space=None,history_candidate_space=None):               ##############在环境中增加历史候选空间
        """Temporal Knowledge Graph Environment.
        examples: quadruples (subject, relation, object, timestamps);
        config: config dict;
        state_action_space: Pre-processed action space;
        history_candidate_space : Pre-processed candidate space;
        """
        self.config = config
        self.num_rel = config['num_rel']
        self.graph, self.label2nodes = self.build_graph(examples)
        ########################################如何构建图是重点################################################
        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        ###############################自环即无操作的实现方式###################################################
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation.
        self.tPAD = 0  # Padding time
        self.window_size=config['window_size']                                                               #####################历史时间窗口大小
        self.state_action_space = state_action_space  # Pre-processed action space
        self.history_candidate_space=history_candidate_space                                                 #############pre-peocessed history space

        if state_action_space:
            self.state_action_space_key = self.state_action_space.keys()
        if history_candidate_space:
            self.history_candidate_space_key=self.history_candidate_space.keys()                             ####################记录key
    def build_graph(self, examples):
        """The graph node is represented as (entity, time), and the edges are directed and labeled relation.
        return:
            graph: nx.MultiDiGraph;
            label2nodes: a dict [keys -> entities, value-> nodes in the graph (entity, time)]
        """
        graph = nx.MultiDiGraph()
        label2nodes = defaultdict(set)
        examples.sort(key=lambda x: x[3], reverse=True)  # Reverse chronological order
        #########################################即顺序从后往前#############################################
        for example in examples:
            src = example[0]
            rel = example[1]
            dst = example[2]
            time = example[3]

            # Add the nodes and edges of the current quadruple
            src_node = (src, time)
            dst_node = (dst, time)
            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)

            graph.add_edge(src_node, dst_node, relation=rel)
            graph.add_edge(dst_node, src_node, relation=rel+self.num_rel+1)

            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)
        return graph, label2nodes

    def get_state_actions_space_complete(self, entity, time, current_=False, max_action_num=None):
        """Get the action space of the current state.
        Args:
            entity: The entity of the current state;
            time: Maximum timestamp for candidate actions;
            current_: Can the current time of the event be used;
            max_action_num: Maximum number of events stored;
        Return:
            numpy array，shape: [number of events，3], (relation, dst, time)
        """
        if self.state_action_space:
            if (entity, time, current_) in self.state_action_space_key:
                return self.state_action_space[(entity, time, current_)]
        nodes = self.label2nodes[entity].copy()
        if current_:
            # Delete future events, you can see current events, before query time
            nodes = list(filter((lambda x: x[1] <= time), nodes))
        else:
            # No future events, no current events
            nodes = list(filter((lambda x: x[1] < time), nodes))
        nodes.sort(key=lambda x: x[1], reverse=True)
        actions_space = []
        i = 0
        for node in nodes:
            for src, dst, rel in self.graph.out_edges(node, data=True):
                actions_space.append((rel['relation'], dst[0], dst[1]))
                i += 1
                if max_action_num and i >= max_action_num:
                    break
            if max_action_num and i >= max_action_num:
                break
        return np.array(list(actions_space), dtype=np.dtype('int32'))

    def get_history_candidate_space(self, head, relation, time, max_candidate_num=None):
        """Get the history candidate space of the query.
        Args:
            head: The head entity of the query;
            realtion: The relation of the query;
            time: The time of the query;
            max_candidate_num: The max candidate history entity of the query;

        Return:
            numpy array，shape: [number of candidate，2],(dst, time)
        """
        if self.history_candidate_space:
            if (head, relation, time) in self.history_candidate_space_key:
                return self.history_candidate_space[(head, relation, time)]
        # 使用key直接获取value 要求在预处理阶段就要保存需要的数据
        nodes = self.label2nodes[head].copy()
        window_left = max(0, time - self.window_size)
        nodes = list(filter((lambda x: window_left <= x[1] < time), nodes))
        nodes.sort(key=lambda x: x[1], reverse=False)

        history_candidates = []
        for node in nodes:
            for src, dst, rel in self.graph.out_edges(node, data=True):
                if rel['relation'] == relation:
                    history_candidates.append((dst[0], dst[1]))
        return np.array(list(history_candidates), dtype=np.dtype('int32'))
    def next_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Get the current action space. There must be an action that stays at the current position in the action space.
        Args:
            entites: torch.tensor, shape: [batch_size], the entity where the agent is currently located;
            times: torch.tensor, shape: [batch_size], the timestamp of the current entity;
            query_times: torch.tensor, shape: [batch_size], the timestamp of query;
            max_action_num: The size of the action space;
            first_step: Is it the first step for the agent.
        Return: torch.tensor, shape: [batch_size, max_action_num, 3], (relation, entity, time)
        """
        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()
        #######################################################################################
        actions = self.get_padd_actions(entites, times, query_times, max_action_num, first_step)
        #######################################################################################
        if self.config['cuda']:
            actions = torch.tensor(actions, dtype=torch.long, device='cuda')
        else:
            actions = torch.tensor(actions, dtype=torch.long)
        return actions

    def get_padd_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Construct the model input array.
        If the optional actions are greater than the maximum number of actions, then sample,
        otherwise all are selected, and the insufficient part is pad.
        """
        actions = np.ones((entites.shape[0], max_action_num, 3), dtype=np.dtype('int32'))
        actions[:, :, 0] *= self.rPAD
        actions[:, :, 1] *= self.ePAD
        actions[:, :, 2] *= self.tPAD
        for i in range(entites.shape[0]):
            # NO OPERATION
            actions[i, 0, 0] = self.NO_OP
            actions[i, 0, 1] = entites[i]
            actions[i, 0, 2] = times[i]

            if times[i] == query_times[i]:

                action_array = self.get_state_actions_space_complete(entites[i], times[i], False)

            else:
                action_array = self.get_state_actions_space_complete(entites[i], times[i], True)
            #######################################################################################
            if action_array.shape[0] == 0:
                continue
            #########################################################################################
            # Whether to keep the action NO_OPERATION
            start_idx = 1
            if first_step:
                # The first step cannot stay in place
                start_idx = 0

            if action_array.shape[0] > (max_action_num - start_idx):
                # Sample. Take the latest events.
                actions[i, start_idx:, ] = action_array[:max_action_num-start_idx]
            else:
                actions[i, start_idx:action_array.shape[0]+start_idx, ] = action_array
        return actions
    def get_history_space(self,entities,relations,times,max_candidate_num=25):
        """
        GET the history candidate space.
        Args:
            entities:torch.tensor,shape:[batch_size].the query entity;
            relations:torch.tensor,shape:[batch_size],the query relation;
            times:torch.tensor,shape:[batch_size],the timestamp of query;
            max_candidate_num:the size of the candidate space;
        Return:torch.tensor,shape:[batch_size,max_candidate_num,2].(entity,time)
        """
        if self.config['cuda']:
            entities=entities.cpu()
            relations=relations.cpu()
            times=times.cpu()
        entities=entities.numpy()
        relations=relations.numpy()
        times=times.numpy()
        candidates=self.get_padd_candidates(entities,relations,times,max_candidate_num)
        if self.config['cuda']:
            candidates=torch.tensor(candidates,dtype=torch.long,device='cuda')
        else:
            candidates=torch.tensor(candidates,dtype=torch.long)
        return candidates
    def get_padd_candidates(self,entities,relations,times,max_candidate_num=25):
        """
        Construct the input array for the transformer
        if the optional candidates are greater than the maximum number of candidates,then sample,[-max_candidate_num:],
        otherwise all are selected and the insufficient prt is pad,    pad firt or pad last??????
        """
        candidates=np.ones((entities.shape[0],max_candidate_num,2),dtype=np.dtype('int32'))
        candidates[:,:,0] *= self.ePAD
        candidates[:,:,1] *= self.tPAD
        for i in range(entities.shape[0]):
            candidate_array=self.get_history_candidate_space(entities[i],relations[i],times[i])
            if candidate_array.shape[0]==0:
                continue
            if candidate_array.shape[0]>max_candidate_num:
                candidates[i,:,]=candidate_array[-max_candidate_num:]
            else:
                candidates[i,0:candidate_array.shape[0],]=candidate_array
        return candidates