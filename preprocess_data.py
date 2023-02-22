import pickle
import os
import argparse
from model.environment import Env
from dataset.baseDataset import baseDataset
import numpy as np
from tqdm import tqdm
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='data/ICEWS14', type=str, help='Path to data.')
    # parser.add_argument('--data_history',default='data/ICEWS14',type=str,help='Path to history')   ############历史数据路径
    # parser.add_argument('--data_frequency',default='data/ICEWS14',type=str,help='Path to frequency')   ############历史数据频率

    parser.add_argument('--outfile', default='state_actions_space.pkl', type=str,
                        help='file to save the preprocessed action data.')
    parser.add_argument('--hisfile',default='history_candidate_space.pkl',type=str,
                        help='file to save the preprocessed history data')                        ############历史重复实体预处理保存文件
    parser.add_argument('--frefile',default='candidate_recent_frequency.pkl',type=str,
                        help='file to save the preprocessed history data')
    parser.add_argument('--store_actions_num', default=0, type=int,
                        help='maximum number of stored neighbors, 0 means store all.')
    parser.add_argument('--store_candidate_num',default=0,type=int,
                        help='maximum number of history candidate,0 means store all.')            #############存储的历史重复实体数量
    args = parser.parse_args()

    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None
    dataset = baseDataset(trainF, testF, statF, validF)
    config = {
        'num_rel': dataset.num_r,
        'num_ent': dataset.num_e,
        'window_size':60
    }
    env = Env(dataset.allQuadruples, config)
    state_actions_space = {}
    entity_frequency={}

    history_candidate_space={}                                                ################################增加历史查询空间，key为（h,r,t)
    timestamps = list(dataset.get_all_timestamps())

    with tqdm(total=len(dataset.allQuadruples)) as bar:
        for (head, rel, tail, t) in dataset.allQuadruples:
            if (head, t, True) not in state_actions_space.keys():
                state_actions_space[(head, t, True)] = env.get_state_actions_space_complete(head, t, True, args.store_actions_num)
                state_actions_space[(head, t, False)] = env.get_state_actions_space_complete(head, t, False, args.store_actions_num)
            if (tail, t, True) not in state_actions_space.keys():
                state_actions_space[(tail, t, True)] = env.get_state_actions_space_complete(tail, t, True, args.store_actions_num)
                state_actions_space[(tail, t, False)] = env.get_state_actions_space_complete(tail, t, False, args.store_actions_num)
            bar.update(1)
    pickle.dump(state_actions_space, open(os.path.join(args.data_dir, args.outfile), 'wb'))

    print('---------------------------------------------开始历史序列提取---------------------------------------------------------------------')
    with tqdm(total=len(dataset.allQuadruples)) as bar:
        for (head, rel, tail, t) in dataset.allQuadruples:
            # if (head, t, True) not in state_actions_space.keys():
            if (head,rel,t) not in history_candidate_space.keys():
                history_candidate_space[(head,rel,t)] = env.get_history_candidate_space(head, rel, t, args.store_candidate_num)
            if (tail,rel+dataset.num_r+1,t) not in history_candidate_space.keys():
                history_candidate_space[(tail,rel+dataset.num_r+1,t)]=env.get_history_candidate_space(tail,rel+dataset.num_r+1,t,args.store_candidate_num)
            bar.update(1)
    pickle.dump(history_candidate_space,open(os.path.join(args.data_dir,args.hisfile),'wb'))
    print('---------------------------------------------历史序列提取完毕--------------------------------------------------------------------------')

    # # with open(os.path.join(args.data_dir,args.hisfile),'rb')as file:
    # #     dict_data=pickle.load(file,encoding='bytes')
    # with tqdm(total=len(dataset.allQuadruples)) as bar:
    #     for (head,rel,tail,t) in dataset.allQuadruples:
    #         if (head,t) not in entity_frequency.keys():
    #             entity_frequency[(head,t)] = 1
    #         else:
    #             entity_frequency[(head,t)]+=1
    #         if (tail,t) not in entity_frequency.keys():
    #             entity_frequency[(tail,t)] = 1
    #         else:
    #             entity_frequency[(tail,t)]+=1
    #         bar.update(1)
    # actions = np.zeros((7129, 8737), dtype=np.dtype('int32'))
    # for i, j in entity_frequency.keys():
    #     actions[i, j] = entity_frequency[(i, j)]
    # pickle.dump(actions, open(os.path.join(args.data_dir, args.frefile), 'wb'))
    # with open(os.path.join(args.data_dir,args.frefile),'rb')as file:
    #     entity_frequency=pickle.load(file,encoding='bytes')



