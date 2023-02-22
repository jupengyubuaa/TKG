import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseDataset, QuadruplesDataset
from model.agent import Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
from model.dirichlet import Dirichlet
import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true',default=True, help='whether to use GPU or not.')
    parser.add_argument('--data_path', type=str, default='data/WIKI', help='Path to data.')
    parser.add_argument('--do_train', action='store_true',default=True,help='whether to train.')
    parser.add_argument('--do_test', action='store_true',help='whether to test.')
    parser.add_argument('--save_path', default='logs', type=str, help='log and model save path.')
    parser.add_argument('--load_model_path', default='logs', type=str, help='trained model checkpoint path.')

    # Train Params
    parser.add_argument('--batch_size', default=512, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=300, type=int, help='max training epochs.')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number used for dataloader.')
    parser.add_argument('--valid_epoch', default=30, type=int, help='validation frequency.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')
    parser.add_argument('--save_epoch', default=15, type=int, help='model saving frequency.')
    parser.add_argument('--clip_gradient', default=10.0, type=float, help='for gradient crop.')
    ###############################################梯度裁剪在哪用了？########################################################
    # Test Params
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='test batch size, it needs to be set to 1 when using IM module.')
    parser.add_argument('--beam_size', default=100, type=int, help='the beam number of the beam search.')
    parser.add_argument('--test_inductive', action='store_true', help='whether to verify inductive inference performance.')
    parser.add_argument('--IM', action='store_true',default=False,help='whether to use IM module.')
    parser.add_argument('--mu', default=0.1, type=float, help='the hyperparameter of IM module.')
    parser.add_argument('--frefile_pkl', default='candidate_recent_frequency.pkl', type=str,
                        help='the file storing the entity_recent_frequency')
    # Agent Params
    parser.add_argument('--ent_dim', default=100, type=int, help='Embedding dimension of the entities')
    parser.add_argument('--rel_dim', default=100, type=int, help='Embedding dimension of the relations')
    parser.add_argument('--state_dim', default=100, type=int, help='dimension of the LSTM hidden state')
    parser.add_argument('--hidden_dim', default=100, type=int, help='dimension of the MLP hidden layer')
    parser.add_argument('--time_dim', default=20, type=int, help='Embedding dimension of the timestamps')
    parser.add_argument('--entities_embeds_method', default='dynamic', type=str,
                        help='representation method of the entities, dynamic or static')
    parser.add_argument('--nhead',default=4,type=int,help='nhead of the TransformerEncoder')
    parser.add_argument('--d_hid',default=2048,type=int,help='the dimension of the feedforward network model')
    parser.add_argument('--dropout',default=0.1,type=float,help='the dropout value')
    parser.add_argument('--n_layers',default=2,type=int,help='the number of sub-encoder-layers in the encoder')

    # Environment Params
    parser.add_argument('--state_actions_path', default='state_actions_space.pkl', type=str,
                        help='the file stores preprocessed candidate action array.')
    parser.add_argument('--history_candidate_path',default='history_candidate_space.pkl',type=str,
                        help='the file stores preprocessed history candidate array')                           ############增加参数-------预处理环境
    ##############################################预处理过程########################################################
    # Episode Params
    parser.add_argument('--window_size',default=1200,type=int,help='the window size of history')
    parser.add_argument('--path_length', default=3, type=int, help='the agent search path length.')
    parser.add_argument('--max_action_num', default=60, type=int, help='the max candidate actions number.')
    parser.add_argument('--max_candidate_num',default=60,type=int,help='the max candidate of history')         ##########增加参数-------最大历史实体数
    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.0, type=float, help='update rate of baseline.')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman Eq.')
    # 策略网络奖励衰减，因为只会再最后给出奖励

    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant.')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term.')

    # reward shaping params
    parser.add_argument('--reward_shaping', action='store_true',help='whether to use reward shaping.')
    parser.add_argument('--time_span', default=24, type=int, help='24 for ICEWS, 1 for WIKI and YAGO')
    parser.add_argument('--alphas_pkl', default='dirchlet_alphas.pkl', type=str,
                        help='the file storing the alpha parameters of the Dirichlet distribution.')
    parser.add_argument('--k', default=300, type=int, help='statistics recent K historical snapshots.')
    ###############################################--k在哪里使用了？？######################################################
    return parser.parse_args(args)

def get_model_config(args, num_ent, num_rel):
    config = {
        'window_size': args.window_size, # window of query
        'nhead': args.nhead,
        'dropout': args.dropout,
        'd_hid': args.d_hid,
        'n_layers': args.n_layers,
        'cuda': args.cuda,  # whether to use GPU or not.
        'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'ent_dim': args.ent_dim,  # Embedding dimension of the entities
        'rel_dim': args.rel_dim,  # Embedding dimension of the relations
        'time_dim': args.time_dim,  # Embedding dimension of the timestamps
        'state_dim': args.state_dim,  # dimension of the LSTM hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        # 'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_input_dim': args.ent_dim*2 + args.rel_dim + args.state_dim,  # 计算得分二
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        'max_candidate_num':args.max_candidate_num,
        'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman Eq.
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used
    }
    return config

def main(args):
    #######################Set Logger#################################
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path = os.path.join(args.data_path, 'train.txt')
    test_path = os.path.join(args.data_path, 'test.txt')
    stat_path = os.path.join(args.data_path, 'stat.txt')
    valid_path = os.path.join(args.data_path, 'valid.txt')

    baseData = baseDataset(train_path, test_path, stat_path, valid_path)

    trainDataset  = QuadruplesDataset(baseData.trainQuadruples, baseData.num_r)

    train_dataloader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    validDataset = QuadruplesDataset(baseData.validQuadruples, baseData.num_r)
    valid_dataloader = DataLoader(
        validDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    testDataset = QuadruplesDataset(baseData.testQuadruples, baseData.num_r)
    test_dataloader = DataLoader(
        testDataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ######################Creat the agent and the environment###############################
    config = get_model_config(args, baseData.num_e, baseData.num_r)
    logging.info(config)
    logging.info(args)

    # creat the agent
    agent = Agent(config)
    ###########################################重点###########################################
    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    history_candidate_path = os.path.join(args.data_path, args.history_candidate_path)                    ##############历史实体路径

    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(os.path.join(args.data_path, args.state_actions_path), 'rb'))
    if not os.path.exists(history_candidate_path):                                                       ###############历史候选空间
        history_candidate_space = None
    else:
        history_candidate_space = pickle.load(open(os.path.join(args.data_path, args.history_candidate_path), 'rb'))
    env = Env(baseData.allQuadruples, config, state_action_space,history_candidate_space)               ################环境构建时增加历史候选空间
    # Create episode controller

    episode = Episode(env, agent,config)
    if args.cuda:
        episode = episode.cuda()
    pg = PG(config)  # Policy Gradient
    #########################################策略网络如何工作#####################################
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    ######################Training and Testing###########################
    if args.reward_shaping:
        alphas = pickle.load(open(os.path.join(args.data_path, args.alphas_pkl), 'rb'))
        distributions = Dirichlet(alphas, args.k)
    else:
        distributions = None
    trainer = Trainer(episode, pg, optimizer, args, distributions)
    tester = Tester(episode, args, baseData.train_entities, baseData.RelEntCooccurrence)
    if args.do_train:
        logging.info('Start Training......')
        for i in range(args.max_epochs):
            loss, reward = trainer.train_epoch(train_dataloader, trainDataset.__len__())
            ########################################计算损失和奖励，为核心部分#############################
            logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

            if i % args.save_epoch == 0 and i != 0:
                trainer.save_model('checkpoint_{}.pth'.format(i))
                logging.info('Save Model in {}'.format(args.save_path))
            ###############################################################################################
            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start Val......')
                metrics = tester.test(valid_dataloader,
                                      validDataset.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))
            if i % args.valid_epoch == 0 and i != 0:
                logging.info('Start test......')
                metrics = tester.test(test_dataloader,
                                      test_dataloader.__len__(),
                                      baseData.skip_dict,
                                      config['num_ent'])
                for mode in metrics.keys():
                    logging.info('{} at epoch {}: {}'.format(mode, i, metrics[mode]))
            ###################################################################################################
        trainer.save_model()
        logging.info('Save Model in {}'.format(args.save_path))

    if args.do_test:
        logging.info('Start Testing......')
        # metrics,metrics_process= tester.test(test_dataloader,
        metrics= tester.test(test_dataloader,
                              testDataset.__len__(),
                              baseData.skip_dict,
                              config['num_ent'])
        for mode in metrics.keys():
            logging.info('Test {} : {}'.format(mode, metrics[mode]))
        # for mode in metrics_process.keys():
        #     logging.info('Test {} : {}'.format(mode, metrics_process[mode]))

if __name__ == '__main__':
    print(torch.__version__)
    args = parse_args()
    main(args)