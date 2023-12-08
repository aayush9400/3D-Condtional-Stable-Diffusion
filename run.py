import argparse
from train_eval import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', default=False, action='store_true', help='Augment Data (FBC)')
    parser.add_argument('--train', action='store_true', help='training flag - VQVAE')
    parser.add_argument('--train_dm', action='store_true', help='training flag - Diffsuion')
    parser.add_argument('--dataset', type=str, default='both', help='options for dataset -> HCP, NFBS, CC, both, all')
    parser.add_argument('--test', action='store_true', help='testing flag - VQVAE')
    parser.add_argument('--test_dm', action='store_true', help='testing flag - Diffsuion')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lbs', type=int, default=5, help='Batch size per gpu')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--val_perc', type=float, default=0.1, help='Validation Percentage of Dataset')
    parser.add_argument('--suffix', default='basic', type=str, help='output or ckpts saved with this suffix')
    parser.add_argument('--num_gpus', default=2, type=int, help='Number of GPUs to be used')
    parser.add_argument('--kernel_resize', action='store_true', help='kernel resize flag')
    parser.add_argument('--test_epoch', type=int)
    parser.add_argument('--save_best_only', default=False, action='store_true')
    parser.add_argument('--vqvae_load_ckpt', type=str, default=None)
    parser.add_argument('--timesteps', type=int, default=300)
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--test_run', default=False, action='store_true')
    args = parser.parse_args()

    run(args)