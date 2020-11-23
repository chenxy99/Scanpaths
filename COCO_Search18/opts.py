from __future__ import print_function
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--mode", type=str, default="train", help="Selecting running mode (default: train)")
    parser.add_argument("--img_dir", type=str, default="./data/images", help="Directory to the image data (stimuli)")
    parser.add_argument("--fix_dir", type=str, default="./data/fixations", help="Directory to the raw fixation file")
    parser.add_argument("--detector_dir", type=str, default="./data/detectors", help="Directory to detector results")
    parser.add_argument("--width", type=int, default=320, help="Width of input data")
    parser.add_argument("--height", type=int, default=240, help="Height of input data")
    parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
    parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
    parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
    parser.add_argument("--detector_threshold", type=float, default=0.8, help="threshold for the detector")
    parser.add_argument("--clip", type=float, default=12.5, help="Gradient clipping")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_epoch", type=int, default=1, help="Epoch when finishing warn up strategy")
    parser.add_argument("--start_rl_epoch", type=int, default=5, help="Epoch when starting reinforcement learning")
    parser.add_argument("--rl_sample_number", type=int, default=5,
                        help="Number of samples used in policy gradient update")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rl_lr_initial_decay", type=float, default=0.5, help="Initial decay of learning rate of rl")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--gpu_ids", type=list, default=[0, 1], help="Used gpu ids")
    parser.add_argument("--log_root", type=str, default="./assets/", help="Log root")
    parser.add_argument("--resume_dir", type=str, default="", help="Resume from a specific directory")
    parser.add_argument("--center_bias", type=bool, default=True, help="Adding center bias or not")
    parser.add_argument("--lambda_1", type=float, default=1, help="Hyper-parameter for duration loss term")
    parser.add_argument("--eval_repeat_num", type=int, default=10, help="Repeat number for evaluation")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
    parser.add_argument("--ablate_attention_info", type=bool, default=False,
                        help="Ablate the attention information or not")
    parser.add_argument("--supervised_save", type=bool, default=True,
                        help="Copy the files before start the policy gradient update")

    # config
    parser.add_argument('--cfg', type=str, default=None,
                        help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    return args
