import torch
import tqdm
import numpy as np
import copy
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

#
# def expmove(y,alpha=0.3):
#     n=len(y)
#     M=np.zeros(n)# 生成空序列，用于存储指数平滑值M
#     M[0]=y[0]# 初始指数平滑值的赋值
#     for i in range(1,len(y)):
#         M[i]=alpha*y[i-1]+(1-alpha)*M[i-1]# 开始预测
#     return M
# def SES(y,alpha=0.5):
#     ss1 = expmove(y,alpha)
#     # 二次指数平滑序列
#     ss2 = expmove(ss1,alpha)
#     y_pred=np.zeros(len(y))
#
#     for i in range(1,len(y)):
#         y_pred[i]=2*ss1[i-1]-ss2[i-1]+alpha/(1-alpha)*(ss1[i-1]-ss2[i-1])
#     # 2023
#     y_pred=2*ss1[-1]-ss2[-1]+alpha/(1-alpha)*(ss1[-1]-ss2[-1])*1
#     return y_pred
# # 预测原时间序列
class Tester(object):
    def __init__(self, model, args, train_entities, RelEntCooccurrence):
        self.model = model
        self.args = args
        self.train_entities = train_entities
        self.RelEntCooccurrence = RelEntCooccurrence



    def get_rank(self, score, answer, entities_space, num_ent):
        """Get the location of the answer, if the answer is not in the array,
        the ranking will be the total number of entities.
        Args:
            score: list, entity score
            answer: int, the ground truth entity
            entities_space: corresponding entity with the score
            num_ent: the total number of entities
        Return: the rank of the ground truth.
        """
        if answer not in entities_space:
            rank = num_ent
        else:
            answer_prob = score[entities_space.index(answer)]
            score_copy=copy.copy(score)
            score_copy.sort(reverse=True)
            rank = score_copy.index(answer_prob) + 1
        return rank

    def test(self, dataloader, ntriple, skip_dict, num_ent):
        """Get time-aware filtered metrics(MRR, Hits@1/3/10).
        Args:
            ntriple: number of the test examples.
            skip_dict: time-aware filter. Get from baseDataset
            num_ent: number of the entities.
        Return: a dict (key -> MRR/HITS@1/HITS@3/HITS@10, values -> float)
        """
        self.model.eval()
        logs = []
        # logs_process=[]
        with torch.no_grad():
            with tqdm.tqdm(total=ntriple, unit='ex') as bar:
                current_time = 0
                cache_IM = {}  # key -> entity, values: list, IM representations of the co-o relations.
                x=0
                y=0
                for src_batch, rel_batch, dst_batch, time_batch in dataloader:
                    batch_size = dst_batch.size(0)

                    if self.args.IM:
                        src = src_batch[0].item()
                        rel = rel_batch[0].item()
                        dst = dst_batch[0].item()
                        time = time_batch[0].item()

                        # representation update
                        if current_time != time:
                            current_time = time
                            for k, v in cache_IM.items():
                                ims = torch.stack(v, dim=0)
                                self.model.agent.update_entity_embedding(k, ims, self.args.mu)
                            cache_IM = {}

                        if src not in self.train_entities and rel in self.RelEntCooccurrence['subject'].keys():
                            im = self.model.agent.get_im_embedding(list(self.RelEntCooccurrence['subject'][rel]))
                            if src in cache_IM.keys():
                                cache_IM[src].append(im)
                            else:
                                cache_IM[src] = [im]

                            # prediction shift
                            self.model.agent.entities_embedding_shift(src, im, self.args.mu)

                    if self.args.cuda:
                        src_batch = src_batch.cuda()
                        rel_batch = rel_batch.cuda()
                        dst_batch = dst_batch.cuda()
                        time_batch = time_batch.cuda()
                    ########################################波束搜索############################################
                    current_entities, beam_prob = \
                        self.model.beam_search(src_batch, time_batch, rel_batch)
                    #########################################################################################
                    if self.args.IM and src not in self.train_entities:
                        # We do this
                        # because events that happen at the same time in the future cannot see each other.
                        self.model.agent.back_entities_embedding(src)

                    if self.args.cuda:
                        current_entities = current_entities.cpu()
                        beam_prob = beam_prob.cpu()

                    current_entities = current_entities.numpy()
                    beam_prob = beam_prob.numpy()

                    MRR = 0
                    for i in range(batch_size):
                        candidate_answers = current_entities[i]
                        candidate_score = beam_prob[i]

                        # sort by score from largest to smallest
                        idx = np.argsort(-candidate_score)
                        candidate_answers = candidate_answers[idx]
                        candidate_score = candidate_score[idx]

                        ################################如何对应到实体的顺序##########################################
                        # remove duplicate entities
                        candidate_answers, idx = np.unique(candidate_answers, return_index=True)
                        candidate_answers = list(candidate_answers)
                        candidate_score = list(candidate_score[idx])

                        src = src_batch[i].item()
                        rel = rel_batch[i].item()
                        dst = dst_batch[i].item()
                        time = time_batch[i].item()

                        # get inductive inference performance.
                        # Only count the results of the example containing new entities.

                        if self.args.test_inductive and src in self.train_entities and dst in self.train_entities:
                            continue

                        filter = skip_dict[(src, rel, time)]  # a set of ground truth entities
                        tmp_entities = candidate_answers.copy()
                        tmp_prob = candidate_score.copy()
                        # time-aware filter
                        for j in range(len(tmp_entities)):
                            if tmp_entities[j] in filter and tmp_entities[j] != dst:
                                candidate_answers.remove(tmp_entities[j])
                                candidate_score.remove(tmp_prob[j])
                        ###################################计算rank的时候考虑历史信息，但无法反向传播##########################
                        ranking_raw = self.get_rank(candidate_score, dst, candidate_answers, num_ent)

                        # score=np.zeros(len(candidate_answers))
                        # for j in range(len(candidate_answers)):
                        #     data = self.entity_frequency[candidate_answers[j], range(time - 10 * 24, time, 24)]
                        # #     # model = SimpleExpSmoothing(data)
                        # #     # model_fit = model.fit()
                        # #     # score[i] = model_fit.predict(len(data), len(data))
                        #     score[j]=SES(data)
                        # #     score[j]=self.entity_frequency[candidate_answers[j], time]
                        #     # print(data)
                        #     # print(score[i])
                        #     # print(self.entity_frequency[(candidate_answers[i], time)])
                        # # candidate_score = np.add(candidate_score,0.1*self.entity_frequency[candidate_answers,time]).tolist()
                        # candidate_score = np.add(candidate_score,0.1*score).tolist()
                        # ranking_mature = self.get_rank(candidate_score, dst, candidate_answers, num_ent)
                        # print(candidate_answers)
                        ##########################################################################################
                        logs.append({
                            'MRR': 1.0 / ranking_raw,
                            'HITS@1': 1.0 if ranking_raw <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking_raw <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking_raw <= 10 else 0.0,
                        })
                        # logs_process.append({
                        #     'MRR': 1.0 / ranking_mature,
                        #     'HITS@1': 1.0 if ranking_mature <= 1 else 0.0,
                        #     'HITS@3': 1.0 if ranking_mature <= 3 else 0.0,
                        #     'HITS@10': 1.0 if ranking_mature <= 10 else 0.0,
                        # })
                        MRR = MRR + 1.0 / ranking_raw

                    bar.update(batch_size)
                    bar.set_postfix(MRR='{}'.format(MRR / batch_size))
        metrics = {}
        metrics_process = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        # for metric_process in logs_process[0].keys():
        #     metrics_process[metric_process] = sum([log_process[metric_process] for log_process in logs_process]) / len(logs_process)
        return metrics
        # return metrics,metrics_process
