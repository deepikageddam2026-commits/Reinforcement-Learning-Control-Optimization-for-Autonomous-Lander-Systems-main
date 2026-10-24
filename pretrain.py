"""Reproduce Part B training with the notebook settings and save weights + logs to models/.

Usage:  python pretrain.py dqn [seed]        (about 1 min on Apple Silicon)
        python pretrain.py reinforce [seed]  (about 4 min)
A non-zero seed is saved with a _seed<N> suffix.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import rl_core as rl

MODELS = Path(__file__).parent / 'models'
MODELS.mkdir(exist_ok=True)


def main(which, seed):
    start = time.time()
    suffix = '' if seed == 0 else f'_seed{seed}'
    if which == 'dqn':
        trainer = rl.train_dqn(rl.DQN_SETTINGS, seed, report_every=50)
    else:
        trainer = rl.train_reinforce(rl.REINFORCE_SETTINGS, seed, report_every=100)

    for agent, log, finished in trainer:
        n = len(log['episode_reward'])
        print(f'{which} episode {n:5d} | step {log["step_at_end"][-1]:8d} | '
              f'avg reward (last 100) {np.mean(log["episode_reward"][-100:]):8.1f} | {time.time() - start:5.0f}s',
              flush=True)

    network = agent.q_network if which == 'dqn' else agent.policy_network
    torch.save(network.state_dict(), MODELS / f'{which}_lunarlander{suffix}.pt')
    np.savez(MODELS / f'{which}_training_log{suffix}.npz', **{k: np.array(v) for k, v in log.items()})
    settings = rl.DQN_SETTINGS if which == 'dqn' else rl.REINFORCE_SETTINGS
    (MODELS / f'{which}_settings{suffix}.json').write_text(json.dumps({**settings, 'seed': seed}, indent=2))
    print(f'saved {which} in {time.time() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
