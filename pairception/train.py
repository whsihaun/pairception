import argparse
import json
import random

import numpy as np
import torch
from tqdm import tqdm

from . import config, parse
from .data import AerialVG, loader
from .evaluate import run
from .losses import losses
from .model import Pairception
from .upstream import require_kernel


def train(seed, device, epochs=config.EPOCHS, max_steps=None):
    if device.type == 'cuda':
        require_kernel()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Pairception(parse.load()).load_detector()
    train_data = AerialVG('train', model.tokenizer, filtered=True)
    val_data = AerialVG('val', model.tokenizer, filtered=True)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.LR_DROP, gamma=config.LR_GAMMA)
    train_loader, val_loader = loader(train_data, shuffle=True), loader(val_data, shuffle=False)

    def predict(samples, captions):
        outputs = model(samples, captions)
        return outputs['scores'], outputs['boxes']

    config.output_dir().mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir().mkdir(parents=True, exist_ok=True)
    best = None
    with (config.output_dir() / f'train_seed{seed}.jsonl').open('w') as log:
        for epoch in range(1, epochs + 1):
            model.eval()
            model.relation_injection_module.train()
            for step, (samples, annotations) in enumerate(tqdm(train_loader, desc=f'epoch {epoch}/{epochs} train',
                                                           dynamic_ncols=True)):
                if max_steps is not None and step >= max_steps:
                    break
                annotations = [{key: value.to(device) if torch.is_tensor(value) else value for key, value in a.items()}
                               for a in annotations]
                outputs = model(samples.to(device), [a['caption'] for a in annotations], [a['hflip'] for a in annotations])
                loss = losses(outputs, annotations)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config.CLIP_MAX_NORM, error_if_nonfinite=True)
                optimizer.step()
            model.eval()
            metrics = run(predict, val_loader, device, f'epoch {epoch}/{epochs} val')
            scheduler.step()
            log.write(json.dumps({'epoch': epoch, 'val': metrics}) + '\n')
            log.flush()
            if best is None or (metrics['top1'], metrics['top5']) > best:
                best = (metrics['top1'], metrics['top5'])
                state = {key: value.cpu() for key, value in model.relation_injection_module.state_dict().items()}
                torch.save({'relation_injection_module': state, 'seed': seed, 'epoch': epoch},
                           config.pairception_checkpoint(seed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    train(args.seed, torch.device(args.device))


if __name__ == '__main__':
    main()
