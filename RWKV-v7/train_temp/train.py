########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1
    parser.add_argument("--grad_clip", default=1.0, type=float) # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--train_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough

    parser.add_argument("--head_size", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    args.epoch_count = args.magic_prime // 40320
    args.epoch_steps = 40320 // args.real_bsz
    assert args.epoch_steps * args.real_bsz == 40320

    if args.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(args.proj_dir):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = ((p.split("-"))[1].split("."))[0]
                if p != "final":
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
        list_p.sort()
        max_p = list_p[-1]
        if len(list_p) > 1:
            args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
        if max_p == -1:
            args.load_model = f"{args.proj_dir}/rwkv-init.pth"
        else:
            args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            if args.warmup_steps < 0:
                args.warmup_steps = 10
        args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["binidx"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0" # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    from src.model import RWKV
    model = RWKV(args)
    ### [S] Custom ###
    from src.model import L2Wrap
    from torch.nn import functional as F
    class CustomRWKV(RWKV):
        def forward(self, idx):
            args = self.args
            x = self.emb(idx)
            v_first = torch.empty_like(x)
            for block in self.blocks:
                if args.grad_cp == 1:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
                else:
                    x, v_first = block(x, v_first)
            x = self.ln_out(x)
            x = self.head(x)  # [16, 512, 4]
            return x
        def training_step(self, batch, _):
            data, targets = batch
            # print(data.shape)
            # print(data)
            # print(targets.shape)
            # print(targets)
            logits = self(data)  # [batch_size, num_classes]
            # print(logits.shape)
            # print(logits)
            batch_size, n_task = targets.shape
            n_label = logits.shape[-1] // n_task
            total_loss = 0
            for task_idx in range(n_task):
                task_target = targets[:, task_idx]  # [batch_size]
                task_logit  = logits[:, task_idx * n_label:(task_idx + 1) * n_label]  # [batch_size, num_classes]
                task_loss   = F.cross_entropy(task_logit, task_target)
                total_loss += task_loss
            return total_loss / n_task
    args.epoch_count = 100
    args.epoch_steps = 100
    model = CustomRWKV(args)
    ### [E] Custom ###

    if len(args.load_model) == 0 or args.train_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.train_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
    # model.load_state_dict(load_dict)

    ### [S] Custom ###
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, csvfile: str, seq_len=512):
            df = pd.read_csv(csvfile).sort_values(by='timestamp', ascending=True)
            self.seq_len   = seq_len
            self.timestamp = df["timestamp"].to_numpy()
            self.cols_data = df.columns[1:np.where(df.columns.str.contains("^gt_"))[0].min()].tolist()
            self.cols_gt   = df.columns[df.columns.str.contains("^gt_(?:min|max)_label_[0-9]+", regex=True)].tolist()
            self.data      = df[self.cols_data].to_numpy().astype(np.float32)
            self.gt        = df[self.cols_gt].to_numpy().astype(np.int64)
            self.indexes   = np.arange(seq_len, len(self.data), seq_len // 8)
        def __len__(self):
            return len(self.indexes)
        def __str__(self):
            return "MyDataset"
        def __getitem__(self, idx: int):
            _idx = self.indexes[idx]
            return torch.tensor(self.data[_idx - self.seq_len:_idx]), torch.tensor(self.gt[_idx])
    train_data = MyDataset("getdata/data.csv", seq_len=512)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    print(data_loader)
    class CustomEmbLayer(torch.nn.Module):
        def __init__(self, input_dim: int, seq_len: int, output_dim: int):
            super().__init__()
            self.input_dim  = input_dim
            self.output_dim = output_dim
            self.batch_norm = torch.nn.BatchNorm1d(input_dim)
            self.layer_norm = torch.nn.LayerNorm(seq_len, elementwise_affine=False)
            self.linear     = torch.nn.Linear(input_dim * 2, output_dim, bias=True)
        def forward(self, input: torch.Tensor):
            outputA = self.layer_norm(input.permute(0, 2, 1)).permute(0, 2, 1)
            outputB = self.batch_norm(input.permute(0, 2, 1)).permute(0, 2, 1)
            output  = torch.cat([outputA, outputB], dim=-1)
            output  = self.linear(output)
            return output
    class CustomHeadLayer(torch.nn.Module):
        def __init__(self, input_size, seq_len: int, n_symbols: int, num_classes: int=5):
            super().__init__()
            self.l1  = torch.nn.Linear(input_size, n_symbols, bias=True)
            self.act = torch.nn.ReLU()
            self.l2  = torch.nn.Linear(seq_len * n_symbols, n_symbols * num_classes, bias=False)
        def forward(self, x):
            x = self.l1(x)
            x = self.act(x)
            x = x.view(x.size(0), -1)
            x = self.l2(x)
            return x
    model.emb  = CustomEmbLayer(len(train_data.cols_data), train_data.seq_len, args.n_embd)
    model.head = CustomHeadLayer(args.n_embd, train_data.seq_len, len(train_data.cols_gt), num_classes=5)  # 5クラス分類用
    print(model)
    ### [E] Custom ###
        
    args.vocab_size = 65536  # 固定値に設定

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    # data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    if trainer.global_rank == 0:
        print(f'### Preparing for training (loaded {args.load_model}). Please wait...')
    trainer.fit(model, data_loader)
