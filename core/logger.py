import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def get_default_path_options():
    return OrderedDict([
        ('exp', 'experiments'),
        ('log', 'logs'),
        ('tb_logger', 'tb_logger'),
        ('results', 'results'),
        ('mat_results', 'mat_results'),
        ('checkpoint', 'checkpoint'),
        ('resume_state', None),
    ])

def get_disabled_path_options():
    return OrderedDict([
        ('experiments_root', None),
        ('exp', None),
        ('log', None),
        ('tb_logger', None),
        ('results', None),
        ('mat_results', None),
        ('checkpoint', None),
        ('resume_state', None),
    ])

def ensure_path_options(opt):
    path_opt = opt.get('path')
    default_path = get_default_path_options()

    if not isinstance(path_opt, dict):
        opt['path'] = get_disabled_path_options()
        return False

    for key, value in default_path.items():
        path_opt.setdefault(key, value)
    return True

def ensure_wandb_options(opt):
    wandb_opt = opt.get('wandb')
    if not isinstance(wandb_opt, dict):
        opt['wandb'] = OrderedDict()
        wandb_opt = opt['wandb']

    wandb_opt.setdefault('project', opt.get('name', 'default'))
    wandb_opt.setdefault('run_name', None)

def parse_json_config(config_path):
    # remove comments starting with '//'
    json_str = ''
    with open(config_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    
    return opt

def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb

    opt = parse_json_config(opt_path)
    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    has_path_options = ensure_path_options(opt)
    ensure_wandb_options(opt)
    if has_path_options:
        experiments_root = os.path.join(
            f"{opt['path']['exp']}/ddpm", '{}_{}'.format(opt['name'], get_timestamp()))
        opt['path']['experiments_root'] = experiments_root
        for key, path in opt['path'].items():
            if 'resume' not in key and 'experiments' not in key:
                if key == 'exp':
                    opt['path'][key] = os.path.join(experiments_root, 'experiments')
                else:
                    opt['path'][key] = os.path.join(experiments_root, path)
                mkdirs(opt['path'][key])

    # change dataset length limit
    opt['phase'] = phase

    # export CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # W&B Logging
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass
    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass
    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass
    opt['enable_wandb'] = enable_wandb
    

    try:
        opt["SLF_config"]["config"]=parse_json_config(opt["SLF_config"]["config-path"])
        opt["SLF_config"]["config"]["path"]["resume_state"] = opt["SLF_config"]["checkpoint"]
    except Exception as e:
        print(e)
        
    try:
        opt["PSD_config"]["config"]=parse_json_config(opt["PSD_config"]["config-path"])
        opt["PSD_config"]["config"]["path"]["resume_state"] = opt["PSD_config"]["checkpoint"]
    except Exception as e:
        print(e)
    
    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    if root is not None:
        mkdirs(root)
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        l.addHandler(fh)
    l.setLevel(level)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
