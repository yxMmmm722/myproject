import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

DEFAULT_ROOT_PATH = './data/ETT/'
DEFAULT_DATA_PATH = 'ETTh1.csv'
DEFAULT_FREQ = 'h'
DEFAULT_TEXT_ENCODER_PATH = './models/bert-base-uncased'

DATA_PRESETS = {
    'ETTh1': {'root_path': './data/ETT/', 'data_path': 'ETTh1.csv', 'freq': 'h'},
    'ETTh2': {'root_path': './data/ETT/', 'data_path': 'ETTh2.csv', 'freq': 'h'},
    'ETTm1': {'root_path': './data/ETT/', 'data_path': 'ETTm1.csv', 'freq': 't'},
    'ETTm2': {'root_path': './data/ETT/', 'data_path': 'ETTm2.csv', 'freq': 't'},
    'etth1': {'root_path': './data/ETT/', 'data_path': 'ETTh1.csv', 'freq': 'h'},
    'etth2': {'root_path': './data/ETT/', 'data_path': 'ETTh2.csv', 'freq': 'h'},
    'ettm1': {'root_path': './data/ETT/', 'data_path': 'ETTm1.csv', 'freq': 't'},
    'ettm2': {'root_path': './data/ETT/', 'data_path': 'ETTm2.csv', 'freq': 't'},
    'electricity': {'root_path': './data/electricity/', 'data_path': 'electricity.csv', 'freq': 'h'},
    'Electricity': {'root_path': './data/electricity/', 'data_path': 'electricity.csv', 'freq': 'h'},
    'exchange_rate': {'root_path': './data/exchange_rate/', 'data_path': 'exchange_rate.csv', 'freq': 'd'},
    'exchange': {'root_path': './data/exchange_rate/', 'data_path': 'exchange_rate.csv', 'freq': 'd'},
    'Exchange': {'root_path': './data/exchange_rate/', 'data_path': 'exchange_rate.csv', 'freq': 'd'},
    'illness': {'root_path': './data/illness/', 'data_path': 'national_illness.csv', 'freq': 'w'},
    'Illness': {'root_path': './data/illness/', 'data_path': 'national_illness.csv', 'freq': 'w'},
    'traffic': {'root_path': './data/traffic/', 'data_path': 'traffic.csv', 'freq': 'h'},
    'Traffic': {'root_path': './data/traffic/', 'data_path': 'traffic.csv', 'freq': 'h'},
    'weather': {'root_path': './data/weather/', 'data_path': 'weather.csv', 'freq': 't'},
    'Weather': {'root_path': './data/weather/', 'data_path': 'weather.csv', 'freq': 't'},
}


def apply_data_preset(args):
    preset = DATA_PRESETS.get(args.data)
    if preset is None:
        return

    # Only auto-fill when user keeps default values.
    if args.root_path == DEFAULT_ROOT_PATH and args.data_path == DEFAULT_DATA_PATH:
        args.root_path = preset['root_path']
        args.data_path = preset['data_path']

    if args.freq == DEFAULT_FREQ:
        args.freq = preset['freq']

#test ss
if __name__ == '__main__':
    fix_seed = 0
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, we currently support only long_term_forecast, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='temp', help='model id')
    parser.add_argument('--model', type=str, default='RAFT',
                        help='model name, options: [RAFT]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default=DEFAULT_ROOT_PATH, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default=DEFAULT_FREQ,
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument(
        '--output_attention', action='store_true',
        help='whether to output attention in ecoder'
    )
    parser.add_argument(
        '--n_period', type=int, default=3,
        help='number of periods from {1, 2, 4}'
    )
    parser.add_argument(
        '--topm', type=int, default=20,
        help='Number of Retrievals'
    )
    parser.add_argument('--retrieval_alpha', type=float, default=0.7,
                        help='fused retrieval score weight for numerical similarity')
    parser.add_argument('--learnable_alpha', action='store_true',
                        help='use learnable alpha for retrieval fusion')
    parser.add_argument('--online_retrieval', action='store_true',
                        help='compute retrieval online instead of using precomputed cache')
    parser.add_argument('--retrieval_coarse_k', type=int, default=80,
                        help='coarse retrieval top-k using waveform similarity')
    parser.add_argument('--context_dim', type=int, default=64,
                        help='context encoder hidden dimension')
    parser.add_argument('--gate_hidden_dim', type=int, default=128,
                        help='hidden dim of candidate gate in retrieval aggregation')
    parser.add_argument('--period_router_hidden_dim', type=int, default=128,
                        help='hidden dim of query-adaptive period fusion router')
    parser.add_argument('--no_gated_aggregation', action='store_true',
                        help='disable gated candidate aggregation and fallback to context-softmax weighting')
    parser.add_argument('--freeze_context_encoder', action='store_true',
                        help='freeze context encoder and only train alpha/predictor')
    parser.add_argument('--no_refresh_context_each_epoch', action='store_true',
                        help='disable refreshing context pool every training epoch (online retrieval only)')
    parser.add_argument('--meta_hidden_dim', type=int, default=128,
                        help='hidden dim of meta context encoder')
    parser.add_argument('--meta_embed_dim', type=int, default=64,
                        help='output dim of meta context encoder')
    parser.add_argument('--text_encoder_name', type=str, default=DEFAULT_TEXT_ENCODER_PATH,
                        help='frozen text encoder name')
    parser.add_argument('--require_text_encoder', action='store_true',
                        help='raise error if transformer text encoder cannot be loaded')
    parser.add_argument('--text_max_len', type=int, default=32,
                        help='max token length for text encoder')
    parser.add_argument('--text_proj_dim', type=int, default=64,
                        help='projected text feature dim for multimodal fusion')
    parser.add_argument('--text_batch_size', type=int, default=64,
                        help='batch size for text encoding during cache building')
    parser.add_argument('--save_meta_texts', action='store_true',
                        help='dump generated meta-texts to disk for inspection')
    parser.add_argument('--meta_text_dump_dir', type=str, default='./meta_text_dumps/',
                        help='output directory for dumped meta-text files')
    parser.add_argument('--meta_text_max_samples', type=int, default=200,
                        help='max number of dumped text samples per split')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='qdf_lite',
                        help='loss function, options:[MSE, qdf_lite]')
    parser.add_argument('--qdf_beta', type=float, default=0.7,
                        help='target mixing ratio for quadratic term in qdf_lite')
    parser.add_argument('--qdf_warmup_epochs', type=int, default=5,
                        help='linear warmup epochs for qdf beta')
    parser.add_argument('--qdf_diff_weight', type=float, default=0.15,
                        help='weight of first-difference consistency loss in qdf_lite')
    parser.add_argument('--qdf_level_weight', type=float, default=0.05,
                        help='weight of sequence level-bias loss in qdf_lite')
    parser.add_argument('--qdf_ema_decay', type=float, default=0.98,
                        help='ema decay for covariance tracking in qdf_lite')
    parser.add_argument('--qdf_bandwidth', type=int, default=32,
                        help='bandwidth for covariance matrix in qdf_lite (<=0 means full)')
    parser.add_argument('--qdf_update_interval', type=int, default=1,
                        help='update interval (steps) for covariance tracking in qdf_lite')
    parser.add_argument('--qdf_eps', type=float, default=1e-5,
                        help='numerical epsilon for qdf_lite covariance stabilization')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--retrieval_cache_device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='where to store precomputed retrieval cache')
    parser.add_argument('--text_cache_device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='where to store precomputed text embedding cache')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=0, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    args = parser.parse_args()
    apply_data_preset(args)
    args.refresh_context_each_epoch = not args.no_refresh_context_each_epoch
    args.use_gated_aggregation = not args.no_gated_aggregation
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        # We support only long term forecasting.
        assert(0)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
