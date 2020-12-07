from argparse import ArgumentParser
import sys, os, pdb
sys.path.insert(0, 'src')
from ffwd_tool import ffwd_to_img, ffwd_different_dimensions, ffwd
from utils import get_img, exists, list_files
BATCH_SIZE = 4
DEVICE = '/gpu:0'
INPATH = 'data/testjpg'

# CHECKPOINT_DIR = 'checkpoints2'
# OUTPATH = 'test_out_ourseltrain'


CHECKPOINT_DIR = 'pretrain/la_muse.ckpt'
OUTPATH = 'test_out_pretrain_la_muse'

CHECKPOINT_DIR = 'pretrain/rain_princess.ckpt'
OUTPATH = 'test_out_pretrain_rain_princess'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', default=CHECKPOINT_DIR)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', default=INPATH)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        default=OUTPATH)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir, 
                    device_t=opts.device, batch_size=opts.batch_size)
        else :
            ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device,
                    batch_size=opts.batch_size)

if __name__ == '__main__':
    main()
