"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import pickle
import argparse

import numpy as np
import torch

from model import GPT

parser = argparse.ArgumentParser(description='tinyGPT training script')
parser.add_argument('--out_dir', type=str, default='out-shakespeare-char', help='output directory')
parser.add_argument('--eval_interval', type=int, default=2000, help='how many iterations between evaluations')
parser.add_argument('--log_interval', type=int, default=1, help='how many iterations between logging')
parser.add_argument('--eval_iters', type=int, default=200, help='how many iterations for each evaluation')
parser.add_argument('--eval_only', action='store_true', help='if true, run evaluation only')
parser.add_argument('--always_save_checkpoint', action='store_false', help='if true, always save a checkpoint after each evaluation')
# data
parser.add_argument('--dataset', type=str, default='shakespeare_char', help='dataset name')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of gradient accumulation steps')
parser.add_argument('--batch_size', type=int, default=12, help='batch size')
parser.add_argument('--block_size', type=int, default=1024, help='block size')
# model
parser.add_argument('--vocab_size', type=int, default=None, help='vocab size, if None will use the vocab size of the dataset')
parser.add_argument('--n_layer', type=int, default=12, help='number of transformer layers')
parser.add_argument('--n_head', type=int, default=12, help='number of attention heads')
parser.add_argument('--n_embd', type=int, default=768, help='embedding size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--bias', action='store_true', help='use bias in layernorm')
# adamw optimizer
parser.add_argument('--learning_rate', type=float, default=6e-4, help='maximum learning rate')
parser.add_argument('--max_iters', type=int, default=600000, help='maximum number of iterations')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adamw')
parser.add_argument('--beta2', type=float, default=0.95, help='beta2 for adamw')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value, 0 means no clipping')
# system
parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu or mps')
parser.add_argument('--dtype', type=str, default='bfloat16', help='float32, bfloat16 or float16')
args = parser.parse_args()

# various inits, derived attributes, I/O setup
tokens_per_iter = args.gradient_accumulation_steps * args.batch_size * args.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

# poor man's data loader
data_dir = os.path.join('data', args.dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(args.device, non_blocking=True)
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y

iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    args.vocab_size = meta['vocab_size']
    print(f"found vocab_size = {args.vocab_size} (inside {meta_path})")

# init a new model from scratch
print("Initializing a new model from scratch")
model = GPT(args)
model.to(args.device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
while True:

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % args.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                print(f"saving checkpoint to {args.out_dir}")
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
    if iter_num == 0 and args.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.gradient_accumulation_steps):
        logits, loss = model(X, Y)
        loss = loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % args.log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * args.gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > args.max_iters:
        break
