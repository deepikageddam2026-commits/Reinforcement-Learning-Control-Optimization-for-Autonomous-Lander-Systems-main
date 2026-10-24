"""Gradio demo for ML Lab case study Q9: RL control for an autonomous lunar lander.

Tabs follow the assessment parts:
  Part A  Environment exploration & random baseline
  Part B  Train DQN / REINFORCE with adjustable hyperparameters (live learning curves)
  Part C  Evaluation, landing videos, robustness to perturbation, agent comparison
"""

import json
import tempfile
import time
from pathlib import Path

import gradio as gr
import gymnasium
import imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import rl_core as rl

ROOT = Path(__file__).parent
MODELS = ROOT / 'models'
OUTPUT = Path(tempfile.gettempdir()) / 'lunar_lander_demo'
OUTPUT.mkdir(exist_ok=True)

DQN_COLOR, REINFORCE_COLOR, RANDOM_COLOR = '#1f3b8f', '#e07b00', '#888888'
OUTCOME_COLORS = {'landed on pad': '#2ca02c', 'landed off pad': '#bcbd22', 'crashed': '#d62728', 'timed out': '#7f7f7f'}
PERTURBATIONS = {
    'None': ('none', [0.0]),
    'Sensor noise, whole flight': ('noise_all', [0.02, 0.05, 0.10]),
    'Sensor noise, touchdown only': ('noise_land', [0.05, 0.10, 0.20]),
    'Sideways gust at touchdown': ('gust', [0.10, 0.25, 0.50]),
    'Tilt kick at touchdown': ('tilt', [0.25, 0.50, 1.00]),
}

plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


# ======================================================================
# Loading pretrained agents ("Load dataset / model")
# ======================================================================
def load_network(kind, path):
    network = rl.QNetwork() if kind == 'dqn' else rl.PolicyNetwork()
    network.load_state_dict(torch.load(path, map_location='cpu'))
    network.eval()
    return network


def load_log(path):
    data = np.load(path)
    return {k: data[k].tolist() for k in data.files}


def load_pretrained():
    # part_b_training_logs.npz is the combined log the notebook saves (dqn_* / reinforce_* keys);
    # if it is present it takes priority over the per-agent logs written by pretrain.py
    combined = MODELS / 'part_b_training_logs.npz'
    combined_logs = {}
    if combined.exists():
        data = load_log(combined)
        for kind in ['dqn', 'reinforce']:
            combined_logs[kind] = {k[len(kind) + 1:]: v for k, v in data.items() if k.startswith(kind + '_')}
    agents = {}
    for kind, label in [('dqn', 'DQN'), ('reinforce', 'REINFORCE')]:
        weights, log = MODELS / f'{kind}_lunarlander.pt', MODELS / f'{kind}_training_log.npz'
        if weights.exists():
            agents[f'Pretrained {label}'] = {'kind': kind, 'network': load_network(kind, weights),
                                             'log': combined_logs.get(kind) or (load_log(log) if log.exists() else None)}
    return agents


PRETRAINED = load_pretrained()


def policy_for(name, agents, sampled=False):
    if name == 'Random policy':
        return rl.random_policy
    agent = agents[name]
    if agent['kind'] == 'dqn':
        return rl.make_dqn_policy(agent['network'])
    return rl.make_reinforce_policy(agent['network'], greedy=not sampled)


def all_agents(session_agents):
    return {**PRETRAINED, **(session_agents or {})}


def agent_choices(session_agents):
    return list(all_agents(session_agents).keys()) + ['Random policy']


# ======================================================================
# Shared plotting / video helpers
# ======================================================================
def save_video(frames, name, fps=50):
    path = OUTPUT / f'{name}_{int(time.time() * 1000)}.mp4'
    imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
    return str(path)


def telemetry_figure(result, title):
    traj = result['trajectory']
    steps = np.arange(len(traj))
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6))
    fig.suptitle(title, fontweight='bold')
    panels = [(0, 'x position', 'X position (0 = pad centre)'), (1, 'y position', 'Altitude'),
              (2, 'x velocity', 'Horizontal velocity'), (3, 'y velocity', 'Vertical velocity'),
              (4, 'angle (deg)', 'Tilt angle')]
    for ax, (i, ylabel, t) in zip(axes.flatten(), panels):
        values = np.degrees(traj[:, i]) if i == 4 else traj[:, i]
        ax.plot(steps, values, color=DQN_COLOR, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
        if i == 0:
            ax.axhspan(-rl.PAD_HALF_WIDTH, rl.PAD_HALF_WIDTH, color='gold', alpha=0.25, label='pad')
            ax.legend(fontsize=8)
        if i == 1:
            ax.axhline(rl.TOUCHDOWN_ALTITUDE, color='crimson', linewidth=0.8, linestyle='--', label='touchdown phase')
            ax.legend(fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.set_ylabel(ylabel)
    ax = axes[1, 2]
    actions = np.array(result['actions'])
    counts = [np.sum(actions == a) for a in range(4)]
    ax.bar(['noop', 'left', 'main', 'right'], counts, color=['#aaaaaa', '#4c9be8', '#e8604c', '#4c9be8'])
    ax.set_title('Actions used (engine firings)', fontsize=10)
    for a in axes[1, :2]:
        a.set_xlabel('step')
    fig.tight_layout()
    return fig


def result_markdown(result):
    icon = {'landed on pad': '✅', 'landed off pad': '🟡', 'crashed': '💥', 'timed out': '⏱️'}[result['outcome']]
    speed = '—' if result['touchdown_speed'] is None else f"{result['touchdown_speed']:.3f}"
    return (f"### {icon} {result['outcome'].upper()}\n"
            f"| Total reward | Steps | Fuel cost | Touchdown vertical speed | Safe touchdown on pad |\n|---|---|---|---|---|\n"
            f"| **{result['reward']:.1f}** | {result['steps']} | {result['fuel']:.1f} | {speed} | "
            f"{'yes' if result['safe_touchdown'] else 'no'} |")


def learning_curve_axes(ax, log, color, label, faint=True):
    steps = np.array(log['step_at_end'])
    rewards = np.array(log['episode_reward'])
    if faint:
        ax.plot(steps, rewards, color=color, alpha=0.18, linewidth=0.6)
    ax.plot(steps, rl.moving_average(rewards), color=color, linewidth=2.2, label=label)


# ======================================================================
# PART A — Environment exploration & random baseline
# ======================================================================
def inspect_environment():
    env = gymnasium.make(rl.ENV_NAME)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()
    states = pd.DataFrame({'Index': range(8), 'State variable': rl.STATE_LABELS,
                           'Low': np.round(obs_space.low.astype(float), 3), 'High': np.round(obs_space.high.astype(float), 3)})
    actions = pd.DataFrame({'Action': list(rl.ACTION_LABELS.keys()), 'Control': list(rl.ACTION_LABELS.values())})
    rewards = pd.DataFrame({
        'Reward component': ['Landing on pad / coming to rest', 'Each leg ground contact', 'Main engine firing',
                             'Side engine firing', 'Crash or leaving screen', 'Shaping (distance, speed, tilt)'],
        'Value': ['+100 (+100 to +140 total)', '+10', '-0.3 per frame', '-0.03 per frame', '-100',
                  'dense, every step'],
    })
    info = (f"**Environment:** `{rl.ENV_NAME}` (Gymnasium {gymnasium.__version__}) — "
            f"observation space `Box(8,)` continuous, action space `Discrete(4)`. "
            f"**Solved** = average reward ≥ 200 over 100 episodes. Episodes are cut at 1,000 steps.")
    return info, states, actions, rewards


def run_random_baseline(n_episodes, seed, progress=gr.Progress()):
    n_episodes, seed = int(n_episodes), int(seed)
    rl.set_seed(seed)
    env = gymnasium.make(rl.ENV_NAME)
    results = []
    for ep in progress.tqdm(range(n_episodes), desc='Random episodes'):
        results.append(rl.run_episode(rl.random_policy, seed + ep, env=env))
    env.close()

    rewards = np.array([r['reward'] for r in results])
    lengths = np.array([r['steps'] for r in results])
    stats = pd.DataFrame({
        'Statistic': ['Episodes', 'Mean reward', 'Std deviation', 'Median reward', 'Min reward', 'Max reward',
                      'Mean episode length', 'Crashed (%)', 'Landed on pad (%)', 'Gap to solve (200)'],
        'Value': [n_episodes, f'{rewards.mean():.2f}', f'{rewards.std():.2f}', f'{np.median(rewards):.2f}',
                  f'{rewards.min():.2f}', f'{rewards.max():.2f}', f'{lengths.mean():.1f} steps',
                  f"{np.mean([r['outcome'] == 'crashed' for r in results]) * 100:.0f}",
                  f"{np.mean([r['outcome'] == 'landed on pad' for r in results]) * 100:.0f}",
                  f'{200 - rewards.mean():.2f}'],
    })

    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig1.suptitle(f'Random policy baseline — reward analysis ({n_episodes} episodes)', fontweight='bold')
    axes[0].bar(range(n_episodes), rewards, color=['tomato' if r < 0 else 'steelblue' for r in rewards], width=1.0)
    axes[0].axhline(rewards.mean(), color='navy', linestyle='--', label=f'mean {rewards.mean():.1f}')
    axes[0].axhline(200, color='green', linestyle=':', label='solve threshold 200')
    axes[0].set(title='Reward per episode', xlabel='episode', ylabel='total reward')
    axes[0].legend(fontsize=8)
    axes[1].hist(rewards, bins=20, color='steelblue', edgecolor='white')
    axes[1].axvline(rewards.mean(), color='navy', linestyle='--', label='mean')
    axes[1].axvline(np.median(rewards), color='darkorange', linestyle='--', label='median')
    axes[1].set(title='Reward distribution', xlabel='total reward', ylabel='count')
    axes[1].legend(fontsize=8)
    axes[2].hist(lengths, bins=20, color='teal', edgecolor='white')
    axes[2].set(title='Episode length distribution', xlabel='steps', ylabel='count')
    fig1.tight_layout()

    shown = results[:5]
    fig2, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig2.suptitle('State trajectories — first 5 episodes (random policy)', fontweight='bold')
    titles = ['X position', 'Y position (altitude)', 'X velocity', 'Y velocity', 'Angle (degrees)', 'Angular velocity']
    for k, r in enumerate(shown):
        traj = r['trajectory']
        for i, ax in enumerate(axes.flatten()):
            values = np.degrees(traj[:, i]) if i == 4 else traj[:, i]
            ax.plot(values, linewidth=1.2, alpha=0.85, label=f'ep {k + 1}')
    for ax, t in zip(axes.flatten(), titles):
        ax.set_title(t, fontsize=10)
        ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
        ax.set_xlabel('step')
    axes[0, 0].legend(fontsize=8)
    fig2.tight_layout()

    fig3, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for k, r in enumerate(shown):
        traj = r['trajectory']
        line, = axes[0].plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=f'ep {k + 1}')
        axes[0].scatter(traj[0, 0], traj[0, 1], color=line.get_color(), marker='o', s=50, zorder=5)
        axes[0].scatter(traj[-1, 0], traj[-1, 1], color=line.get_color(), marker='X', s=70, zorder=5)
    axes[0].axvspan(-0.2, 0.2, color='gold', alpha=0.25, label='landing pad')
    axes[0].set(title='2D flight paths (● start, ✕ end)', xlabel='x', ylabel='altitude')
    axes[0].legend(fontsize=8)
    components = ['Landing\n(+100–140)', 'Legs\n(+10 each)', 'Main engine\n(−0.3/step)', 'Side engine\n(−0.03/step)', 'Crash\n(−100)']
    values = [120, 20, -30, -3, -100]
    axes[1].bar(components, values, color=['#2ca02c', '#8fbc8f', '#ff7f0e', '#ffbb78', '#d62728'])
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set(title='Reward component structure (illustrative episode totals)', ylabel='reward')
    fig3.tight_layout()

    return stats, fig1, fig2, fig3


def watch_random_episode(seed):
    result = rl.run_episode(rl.random_policy, int(seed), render=True)
    return save_video(result['frames'], 'random'), result_markdown(result), telemetry_figure(result, 'Random policy — telemetry')


# ======================================================================
# PART B — Training with adjustable hyperparameters
# ======================================================================
def schedule_preview(algo, total, lr_start, lr_end, gamma_start, gamma_end, eps_start, eps_end, eps_decay, ent_start, ent_end):
    progress = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    gammas = [rl.gamma_schedule(p, gamma_start, gamma_end) for p in progress]
    lrs = [rl.lr_schedule(p, lr_start, max(lr_end, 1e-8)) for p in progress]
    axes[0].plot(progress * 100, gammas, color='navy')
    axes[0].set(title='Discount factor γ', xlabel='% of training')
    axes[1].plot(progress * 100, lrs, color='darkgreen')
    axes[1].set(title='Learning rate', xlabel='% of training')
    if algo == 'DQN':
        steps = progress * total
        axes[2].plot(steps, [rl.epsilon_schedule(s, eps_start, eps_end, eps_decay) for s in steps], color='crimson')
        axes[2].set(title='Exploration ε (epsilon-greedy)', xlabel='environment steps')
    else:
        axes[2].plot(progress * 100, [rl.entropy_schedule(p, ent_start, max(ent_end, 1e-8)) for p in progress], color='purple')
        axes[2].set(title='Entropy bonus coefficient', xlabel='% of training')
    fig.tight_layout()
    return fig


def training_figure(algo, log, total):
    color = DQN_COLOR if algo == 'DQN' else REINFORCE_COLOR
    steps = np.array(log['step_at_end'])
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5))
    ax = axes[0, 0]
    learning_curve_axes(ax, log, color, '100-episode moving average')
    ax.axhline(200, color='green', linestyle=':', label='solved = 200')
    ax.set(title='Learning curve', xlabel='environment steps', ylabel='episode reward', ylim=(-450, 330))
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, rl.moving_average(np.array(log['landed'], dtype=float)) * 100, color=color, linewidth=2)
    ax.set(title='Landing rate (came to rest, 100-episode window)', xlabel='environment steps', ylabel='% episodes', ylim=(-3, 103))

    ax = axes[1, 0]
    if algo == 'DQN':
        ax.plot(steps, log['epsilon'], color='crimson', label='ε')
    else:
        ax.plot(steps, np.array(log['entropy_coef']) / max(log['entropy_coef'][0], 1e-12), color='purple', label='entropy coef (relative)')
    ax.plot(steps, np.array(log['lr']) / log['lr'][0], color='darkgreen', label='learning rate (relative)')
    ax2 = ax.twinx()
    ax2.plot(steps, log['gamma'], color='navy', linestyle='--', label='γ')
    ax2.set_ylabel('γ')
    ax.set(title='Schedules actually applied', xlabel='environment steps')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')

    ax = axes[1, 1]
    loss = np.array(log['loss'], dtype=float)
    ok = ~np.isnan(loss)
    if ok.any():
        ax.plot(steps[ok], loss[ok], color='gray', alpha=0.35, linewidth=0.7)
        ax.plot(steps[ok], rl.moving_average(loss[ok], 20), color='black', linewidth=1.5)
    ax.set(title='Training loss (Huber TD loss)' if algo == 'DQN' else 'Policy-gradient loss', xlabel='environment steps')
    fig.tight_layout()
    return fig


def train_agent(algo, total, seed, lr_start, lr_end, gamma_start, gamma_end, eps_start, eps_end, eps_decay,
                batch_size, target_update, buffer_size, ent_start, ent_end, eps_per_update, save_name, session_agents):
    session_agents = dict(session_agents or {})
    if algo == 'DQN':
        settings = {**rl.DQN_SETTINGS, 'total_steps': int(total), 'lr_start': lr_start, 'lr_end': lr_end,
                    'gamma_start': gamma_start, 'gamma_end': gamma_end, 'eps_start': eps_start, 'eps_end': eps_end,
                    'eps_decay_steps': int(eps_decay), 'batch_size': int(batch_size),
                    'target_update_every': int(target_update), 'buffer_size': int(buffer_size)}
        trainer = rl.train_dqn(settings, int(seed), report_every=10)
    else:
        settings = {**rl.REINFORCE_SETTINGS, 'total_episodes': int(total), 'lr_start': lr_start, 'lr_end': lr_end,
                    'gamma_start': gamma_start, 'gamma_end': gamma_end, 'ent_start': ent_start, 'ent_end': ent_end,
                    'episodes_per_update': int(eps_per_update)}
        trainer = rl.train_reinforce(settings, int(seed), report_every=5)

    start = time.time()
    last_plot = 0.0
    name = (save_name or '').strip() or f'My {algo} ({time.strftime("%H:%M:%S")})'
    unit = 'steps' if algo == 'DQN' else 'episodes'
    yield (training_status_html(algo, 0.0, 0.0, None, 0, 0, None, None, f'Starting… (0 / {int(total):,} {unit})'),
           gr.skip(), session_agents, None)
    for agent, log, finished in trainer:
        n = len(log['episode_reward'])
        step = log['step_at_end'][-1]
        done_frac = min(step / total if algo == 'DQN' else n / total, 1.0)
        elapsed = time.time() - start
        eta = None if finished or done_frac <= 0 else elapsed / done_frac * (1 - done_frac)
        avg = float(np.mean(log['episode_reward'][-100:]))
        landing = float(np.mean(log['landed'][-100:]) * 100)
        if finished:
            network = agent.q_network if algo == 'DQN' else agent.policy_network
            network.eval()
            session_agents[name] = {'kind': algo.lower(), 'network': network, 'log': {k: list(v) for k, v in log.items()}}
            weights_path = OUTPUT / f"{name.replace(' ', '_').replace(':', '-')}.pt"
            torch.save(network.state_dict(), weights_path)
            note = f'✅ Finished — saved as <b>{name}</b>, available in the Part C tabs. Download the weights below.'
            fig = training_figure(algo, log, total)
            yield training_status_html(algo, 1.0, elapsed, None, n, step, avg, landing, note), fig, session_agents, str(weights_path)
            plt.close(fig)
        elif time.time() - last_plot > 1.5:
            last_plot = time.time()
            fig = training_figure(algo, log, total)
            yield training_status_html(algo, done_frac, elapsed, eta, n, step, avg, landing), fig, session_agents, None
            plt.close(fig)
        else:
            yield training_status_html(algo, done_frac, elapsed, eta, n, step, avg, landing), gr.skip(), session_agents, None


def format_clock(seconds):
    seconds = int(round(seconds))
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def training_status_html(algo, fraction, elapsed, eta, episodes, steps, avg_reward, landing_rate, note=None):
    color = DQN_COLOR if algo == 'DQN' else REINFORCE_COLOR
    finished = fraction >= 1.0 and eta is None and note is not None and note.startswith('✅')
    stats = [('Elapsed', format_clock(elapsed)),
             ('Remaining', '—' if eta is None else f'~{format_clock(eta)}'),
             ('Episode', f'{episodes:,}'), ('Env step', f'{steps:,}'),
             ('Avg reward (last 100)', '—' if avg_reward is None else f'{avg_reward:.1f}'),
             ('Landing rate (last 100)', '—' if landing_rate is None else f'{landing_rate:.0f}%')]
    cells = ''.join(f'<div style="flex:1;min-width:110px"><div style="font-size:12px;opacity:.7">{label}</div>'
                    f'<div style="font-size:20px;font-weight:700;font-variant-numeric:tabular-nums">{value}</div></div>'
                    for label, value in stats)
    title = f"{'Finished' if finished else 'Training'} {algo}"
    return (f'<div style="border:1px solid {color};border-radius:10px;padding:12px 14px">'
            f'<div style="display:flex;justify-content:space-between;font-weight:700;margin-bottom:6px">'
            f'<span>{title}</span><span style="font-variant-numeric:tabular-nums">{fraction * 100:.1f}%</span></div>'
            f'<div style="height:12px;border-radius:6px;background:rgba(127,127,127,.25);overflow:hidden">'
            f'<div style="height:100%;width:{fraction * 100:.2f}%;background:{color};transition:width .3s"></div></div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px">{cells}</div>'
            + (f'<div style="margin-top:8px">{note}</div>' if note else '') + '</div>')


def upload_weights(file, kind, session_agents):
    session_agents = dict(session_agents or {})
    if file is None:
        return 'Choose a `.pt` file first.', session_agents
    kind_key = 'dqn' if kind == 'DQN (Q-network)' else 'reinforce'
    try:
        network = load_network(kind_key, file)
    except Exception as error:
        return f'❌ Could not load weights: `{error}`', session_agents
    name = f'Uploaded {Path(file).stem}'
    session_agents[name] = {'kind': kind_key, 'network': network, 'log': None}
    return f'✅ Loaded **{name}** — select it in the Part C tabs.', session_agents


def algo_visibility(algo):
    is_dqn = algo == 'DQN'
    d, r = rl.DQN_SETTINGS, rl.REINFORCE_SETTINGS
    return (gr.update(visible=is_dqn), gr.update(visible=not is_dqn),
            gr.update(label='Total environment steps' if is_dqn else 'Total episodes',
                      minimum=10000 if is_dqn else 100, maximum=400000 if is_dqn else 4000,
                      step=10000 if is_dqn else 100, value=d['total_steps'] if is_dqn else r['total_episodes']),
            gr.update(value=d['lr_start'] if is_dqn else r['lr_start']),
            gr.update(value=d['lr_end'] if is_dqn else r['lr_end']),
            gr.update(value=d['gamma_start'] if is_dqn else r['gamma_start']),
            gr.update(value=d['gamma_end'] if is_dqn else r['gamma_end']))


# ======================================================================
# PART C — Evaluation, robustness and comparison
# ======================================================================
def perturbation_of(family, level):
    kind = PERTURBATIONS[family][0]
    return None if kind == 'none' else (kind, float(level))


def update_levels(family):
    levels = PERTURBATIONS[family][1]
    return gr.update(choices=levels, value=levels[-1] if len(levels) > 1 else levels[0], interactive=len(levels) > 1)


def watch_agent(agent_name, sampled, seed, family, level, session_agents):
    agents = all_agents(session_agents)
    policy = policy_for(agent_name, agents, sampled)
    result = rl.run_episode(policy, int(seed), perturbation_of(family, level), render=True)
    label = agent_name + (' (sampled)' if sampled else '')
    title = f'{label} — seed {int(seed)}' + ('' if family == 'None' else f' — {family} {level}')
    return save_video(result['frames'], 'agent'), result_markdown(result), telemetry_figure(result, title)


def evaluate_agents(agent_names, sampled, n_episodes, family, level, session_agents, progress=gr.Progress()):
    if not agent_names:
        raise gr.Error('Pick at least one agent.')
    agents = all_agents(session_agents)
    seeds = list(range(20000, 20000 + int(n_episodes)))   # unseen start conditions, same as notebook Part C
    perturbation = perturbation_of(family, level)
    summaries, all_results = {}, {}
    for name in agent_names:
        label = name + (' (sampled)' if sampled and name != 'Random policy' and agents.get(name, {}).get('kind') == 'reinforce' else '')
        results = []
        env = gymnasium.make(rl.ENV_NAME)
        policy = policy_for(name, agents, sampled)
        for seed in progress.tqdm(seeds, desc=label):
            results.append(rl.run_episode(policy, seed, perturbation, env=env))
        env.close()
        all_results[label] = results
        summaries[label] = rl.summarise(results)

    rows = []
    for label, s in summaries.items():
        n = s['n']
        landed = round(s['landed on pad'] * n)
        low, high = rl.wilson_interval(landed, n)
        rows.append({'Agent': label, 'Mean reward': round(s['mean_reward'], 1),
                     'Landed on pad %': round(s['landed on pad'] * 100), '95% CI': f'{low * 100:.0f}–{high * 100:.0f}%',
                     'Safe touchdown %': round(s['safe touchdown'] * 100),
                     'Landed off pad %': round(s['landed off pad'] * 100), 'Crashed %': round(s['crashed'] * 100),
                     'Timed out %': round(s['timed out'] * 100), 'Mean steps': round(s['mean_steps']),
                     'Mean fuel cost': round(s['mean_fuel'], 1),
                     'Touchdown speed': None if np.isnan(s['mean_touchdown_speed']) else round(s['mean_touchdown_speed'], 3)})
    table = pd.DataFrame(rows)

    labels = list(summaries.keys())
    short = [l.replace('Pretrained ', '').replace(' (sampled)', '\n(sampled)') for l in labels]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    left = np.zeros(len(labels))
    for outcome in rl.OUTCOMES:
        shares = np.array([summaries[l][outcome] * 100 for l in labels])
        axes[0].barh(short, shares, left=left, color=OUTCOME_COLORS[outcome], label=outcome)
        left += shares
    axes[0].set(title='How the episodes ended', xlabel=f'% of {int(n_episodes)} episodes')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    axes[0].invert_yaxis()
    axes[1].boxplot([[r['reward'] for r in all_results[l]] for l in labels], tick_labels=short)
    axes[1].axhline(200, color='green', linestyle=':')
    axes[1].set_title('Episode reward')
    axes[2].boxplot([[r['fuel'] for r in all_results[l]] for l in labels], tick_labels=short)
    axes[2].set_title('Fuel cost per episode')
    speeds = [[r['touchdown_speed'] for r in all_results[l] if r['touchdown_speed'] is not None] or [np.nan] for l in labels]
    axes[3].boxplot(speeds, tick_labels=short)
    axes[3].set_title('Vertical speed at first leg contact')
    fig.suptitle('Evaluation on unseen start conditions' + ('' if family == 'None' else f' — {family}, level {level}'), fontweight='bold')
    fig.tight_layout()
    return table, fig


def robustness_sweep(agent_names, family, n_episodes, session_agents, progress=gr.Progress()):
    if family == 'None':
        raise gr.Error('Pick a perturbation family.')
    if not agent_names:
        raise gr.Error('Pick at least one agent.')
    agents = all_agents(session_agents)
    kind, levels = PERTURBATIONS[family]
    seeds = list(range(20000, 20000 + int(n_episodes)))
    x_values = [0.0] + levels
    data = {}
    jobs = [(name, lvl) for name in agent_names for lvl in x_values]
    env = gymnasium.make(rl.ENV_NAME)
    for name, lvl in progress.tqdm(jobs, desc='Sweep'):
        policy = policy_for(name, agents)
        perturbation = None if lvl == 0.0 else (kind, lvl)
        results = [rl.run_episode(policy, s, perturbation, env=env) for s in seeds]
        data[(name, lvl)] = rl.summarise(results)
    env.close()

    palette = [DQN_COLOR, REINFORCE_COLOR, '#2ca02c', '#9467bd', '#8c564b', RANDOM_COLOR]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    rows = []
    for color, name in zip(palette, agent_names):
        safe = [data[(name, l)]['safe touchdown'] * 100 for l in x_values]
        rest = [data[(name, l)]['landed on pad'] * 100 for l in x_values]
        err_lo, err_hi = [], []
        for l, rate in zip(x_values, safe):
            lo, hi = rl.wilson_interval(round(rate / 100 * int(n_episodes)), int(n_episodes))
            err_lo.append(rate - lo * 100)
            err_hi.append(hi * 100 - rate)
        short = name.replace('Pretrained ', '')
        axes[0].errorbar(x_values, safe, yerr=[err_lo, err_hi], marker='o', capsize=3, color=color, label=f'{short}: safe touchdown')
        axes[0].plot(x_values, rest, marker='s', linestyle='--', color=color, alpha=0.6, label=f'{short}: at rest on pad')
        axes[1].errorbar(x_values, [data[(name, l)]['mean_reward'] for l in x_values],
                         yerr=[data[(name, l)]['se_reward'] for l in x_values], marker='o', capsize=3, color=color, label=short)
        axes[2].plot(x_values, [data[(name, l)]['mean_fuel'] for l in x_values], marker='o', color=color, label=short)
        for l in x_values:
            s = data[(name, l)]
            rows.append({'Agent': name, 'Level': l, 'At rest on pad %': round(s['landed on pad'] * 100),
                         'Safe touchdown %': round(s['safe touchdown'] * 100), 'Crashed %': round(s['crashed'] * 100),
                         'Timed out %': round(s['timed out'] * 100), 'Mean reward': round(s['mean_reward'], 1),
                         'Mean fuel': round(s['mean_fuel'], 1)})
    axes[0].set(title='Landing success vs perturbation', ylabel='% of episodes', ylim=(-5, 105))
    axes[1].set(title='Mean reward (± 1 s.e.)')
    axes[2].set(title='Mean fuel cost')
    for ax in axes:
        ax.set_xlabel('perturbation level (observation units)')
        ax.legend(fontsize=7)
    fig.suptitle(f'Robustness sweep — {family} ({int(n_episodes)} episodes per point)', fontweight='bold')
    fig.tight_layout()
    return pd.DataFrame(rows), fig


def compare_learning(agent_names, session_agents):
    agents = all_agents(session_agents)
    with_logs = [n for n in agent_names if n in agents and agents[n]['log'] is not None]
    if not with_logs:
        raise gr.Error('None of the selected agents has a training log (uploaded weights have none).')
    palette = [DQN_COLOR, REINFORCE_COLOR, '#2ca02c', '#9467bd', '#8c564b']
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    rows = []
    for color, name in zip(palette, with_logs):
        log = agents[name]['log']
        short = name.replace('Pretrained ', '')
        learning_curve_axes(axes[0], log, color, short)
        steps = np.array(log['step_at_end'])
        avg = rl.moving_average(log['episode_reward'])
        axes[1].plot(steps, avg, color=color, linewidth=2.2, label=short)
        full = np.array(log['episode_reward'])
        full_avg = np.convolve(full, np.ones(100) / 100, mode='valid') if len(full) >= 100 else np.array([])
        full_steps = steps[99:]

        def reach(t):
            idx = np.where(full_avg >= t)[0]
            return f'{int(full_steps[idx[0]]):,}' if len(idx) else 'not reached'

        last = full[-100:]
        rows.append({'Agent': name, 'Episodes': len(full), 'Env steps': f'{int(steps[-1]):,}',
                     'Steps to avg ≥ 0': reach(0), 'Steps to avg ≥ 100': reach(100), 'Steps to avg ≥ 200': reach(200),
                     'Final 100-ep avg': round(float(last.mean()), 1), 'Std last 100': round(float(last.std()), 1),
                     'Worst last 100': round(float(last.min()), 1),
                     'Landing rate last 100 %': round(float(np.mean(log['landed'][-100:]) * 100))})
    for ax in axes:
        ax.axhline(200, color='green', linestyle=':', label='solved = 200')
        ax.set(xlabel='environment steps', ylabel='episode reward')
        ax.legend(loc='lower right', fontsize=8)
    axes[0].set(title='Learning curves, full runs (100-episode moving average)', ylim=(-350, 320))
    axes[1].set(title='Same curves on an equal 200k-step budget', xlim=(0, 200000), ylim=(-250, 300))
    fig.tight_layout()
    return pd.DataFrame(rows), fig


def refresh_agent_lists(session_agents):
    choices = agent_choices(session_agents)
    learned = [c for c in choices if c != 'Random policy']
    return (gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=learned),
            gr.update(choices=learned))


# ======================================================================
# UI
# ======================================================================
INTRO = """
# 🚀 Autonomous Lunar Lander — Reinforcement Learning Control
**ML Lab (MACSE504) · Case Study Q9 · Assessments 8, 9, 10**

*Industry scenario:* an aerospace robotics research centre wants to benchmark RL algorithms that land a module safely on a designated pad with minimal fuel.
Environment: Gymnasium **LunarLander-v3** (current version of v2: same 8 continuous states, 4 discrete actions and rewards).
"""

INFERENCES = """
## Key findings (from the submitted notebook, Parts B & C)

*The numbers below come from the notebook run. The agents bundled with this demo were retrained locally with the same settings and seed, so their live numbers differ slightly. In particular, the local REINFORCE policy hovers when run greedily, which is the "keeps firing, never settles" behaviour described in C.2.*

| | **DQN** (value-based) | **REINFORCE** (policy gradient) |
|---|---|---|
| Steps to reach 100-ep avg ≥ 100 | ~60k | ~837k (≈14× more) |
| Final 100-episode average | ~233 (solved) | ~119 (never reached 200) |
| Landed at rest on pad (100 unseen starts) | 92% | 61% greedy · 0% sampled |
| Mean fuel cost per episode | 45.7 | 121.1 |
| Mean touchdown vertical speed | 0.085 | 0.033 (softer) |

**Sample efficiency.** DQN reuses every transition many times from its replay buffer; REINFORCE uses each batch of 4 episodes once, then discards it.

**Stability.** Neither run collapsed. REINFORCE has lower reward spread but sits on a much lower plateau; DQN occasionally has a bad episode.

**Robustness (C.3).**
- *Gusts:* DQN is stronger at every level (safe touchdown 97 → 75% vs 82 → 46%).
- *Tilt kicks:* similar at moderate levels; a 1.0 kick (2.5 rad/s) is beyond both.
- *Sensor noise:* this is DQN's weak point. Even 0.02 noise stops it from settling: it stands on the pad and keeps firing, so episodes time out. REINFORCE tolerates noise better but uses much more fuel.

**Engineering inferences.**
1. With clean sensing, deploy **DQN**: it lands more often with about 60% less fuel.
2. With noisy sensing, neither policy is ready as is. Filter the state (low-pass or Kalman), train with noise and kicks injected, and add an engine cut-off once both legs are down and speed is near zero.
3. Run REINFORCE **greedily** when deployed: its sampled policy never comes to rest on the pad.

*Limits:* one training seed per algorithm, 100 test episodes per condition (CI ≈ ±10 points), single-kick perturbations.
"""

CSS = """
.gradio-container {max-width: 1400px !important}
#title-md h1 {margin-bottom: 0}
"""


def build_app():
    d, r = rl.DQN_SETTINGS, rl.REINFORCE_SETTINGS
    initial_choices = agent_choices({})
    learned = [c for c in initial_choices if c != 'Random policy']

    with gr.Blocks(title='Lunar Lander RL — Case Study Q9') as demo:
        session_agents = gr.State({})
        gr.Markdown(INTRO, elem_id='title-md')

        with gr.Tabs():
            # ---------------------------------------------------------- PART A
            with gr.Tab('Part A · Environment & Baseline'):
                gr.Markdown('### A.1 Load the environment and inspect its state and action spaces')
                load_env_btn = gr.Button('Load LunarLander-v3 environment', variant='primary')
                env_info = gr.Markdown()
                with gr.Row():
                    state_table = gr.Dataframe(label='Continuous state space (8 dims)', interactive=False)
                    with gr.Column():
                        action_table = gr.Dataframe(label='Discrete actions (4)', interactive=False)
                        reward_table = gr.Dataframe(label='Reward structure', interactive=False)

                gr.Markdown('### A.2 Random control policy baseline')
                with gr.Row():
                    n_random = gr.Slider(10, 300, value=100, step=10, label='Number of episodes')
                    seed_random = gr.Number(value=0, precision=0, label='Seed')
                    run_random_btn = gr.Button('Run random baseline', variant='primary')
                random_stats = gr.Dataframe(label='Baseline reward statistics', interactive=False)
                random_fig1 = gr.Plot(label='Reward analysis')
                gr.Markdown('### A.3 State trajectories, flight paths and reward structure')
                random_fig2 = gr.Plot(label='State trajectories')
                random_fig3 = gr.Plot(label='Flight paths')

                gr.Markdown('### Watch a random-policy episode')
                with gr.Row():
                    watch_seed_a = gr.Number(value=3, precision=0, label='Episode seed')
                    watch_random_btn = gr.Button('Render episode')
                with gr.Row():
                    random_video = gr.Video(label='Random policy', autoplay=True, height=320)
                    random_result = gr.Markdown()
                random_telemetry = gr.Plot(label='Telemetry')

                load_env_btn.click(inspect_environment, None, [env_info, state_table, action_table, reward_table])
                run_random_btn.click(run_random_baseline, [n_random, seed_random], [random_stats, random_fig1, random_fig2, random_fig3])
                watch_random_btn.click(watch_random_episode, watch_seed_a, [random_video, random_result, random_telemetry])

            # ---------------------------------------------------------- PART B
            with gr.Tab('Part B · Train Agents'):
                gr.Markdown('### Train a DQN or REINFORCE agent. Adjust the hyperparameters and schedules, then watch it learn live.\n'
                            'Defaults are the exact settings from the notebook. A full 200k-step DQN run takes about a minute on an Apple Silicon laptop and a few minutes on a free cloud CPU.')
                with gr.Row():
                    with gr.Column(scale=1):
                        algo = gr.Radio(['DQN', 'REINFORCE'], value='DQN', label='Algorithm')
                        total = gr.Slider(10000, 400000, value=d['total_steps'], step=10000, label='Total environment steps')
                        seed_train = gr.Number(value=0, precision=0, label='Seed')
                        with gr.Accordion('Learning rate & discount factor (B.3)', open=True):
                            lr_start = gr.Number(value=d['lr_start'], label='Learning rate — start')
                            lr_end = gr.Number(value=d['lr_end'], label='Learning rate — end (exponential decay)')
                            gamma_start = gr.Slider(0.90, 0.999, value=d['gamma_start'], step=0.001, label='γ start')
                            gamma_end = gr.Slider(0.90, 0.999, value=d['gamma_end'], step=0.001, label='γ end (linear ramp over first 40%)')
                        with gr.Group(visible=True) as dqn_group:
                            eps_start = gr.Slider(0.1, 1.0, value=d['eps_start'], step=0.05, label='ε start')
                            eps_end = gr.Slider(0.0, 0.3, value=d['eps_end'], step=0.01, label='ε end')
                            eps_decay = gr.Slider(2000, 100000, value=d['eps_decay_steps'], step=1000, label='ε decay constant (steps)')
                            batch_size = gr.Dropdown([32, 64, 128, 256], value=d['batch_size'], label='Mini-batch size')
                            target_update = gr.Slider(50, 2000, value=d['target_update_every'], step=50, label='Target network copy every N steps')
                            buffer_size = gr.Dropdown([10000, 50000, 100000, 200000], value=d['buffer_size'], label='Replay buffer size')
                        with gr.Group(visible=False) as reinforce_group:
                            ent_start = gr.Number(value=r['ent_start'], label='Entropy bonus — start')
                            ent_end = gr.Number(value=r['ent_end'], label='Entropy bonus — end')
                            eps_per_update = gr.Slider(1, 16, value=r['episodes_per_update'], step=1, label='Episodes per policy update')
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            save_name = gr.Textbox(label='Save trained agent as', placeholder='e.g. DQN fast decay', scale=3)
                            train_btn = gr.Button('▶ Train', variant='primary', scale=1, size='lg')
                            stop_btn = gr.Button('■ Stop', variant='stop', scale=1, size='lg')
                        train_status = gr.HTML(training_status_html('DQN', 0.0, 0.0, None, 0, 0, None, None, 'Press <b>▶ Train</b> to start.'))
                        train_plot = gr.Plot(label='Live training')
                        schedule_plot = gr.Plot(label='Hyperparameter schedules (preview)')
                        weights_file = gr.File(label='Trained weights (.pt)')
                        with gr.Accordion('Load your own trained weights (.pt)', open=False):
                            upload_kind = gr.Radio(['DQN (Q-network)', 'REINFORCE (policy network)'], value='DQN (Q-network)', label='Network type')
                            upload = gr.File(label='Weights file', file_types=['.pt'], type='filepath')
                            upload_btn = gr.Button('Load weights')
                            upload_status = gr.Markdown()

                schedule_inputs = [algo, total, lr_start, lr_end, gamma_start, gamma_end, eps_start, eps_end, eps_decay, ent_start, ent_end]
                for component in schedule_inputs:
                    component.change(schedule_preview, schedule_inputs, schedule_plot)
                algo.change(algo_visibility, algo, [dqn_group, reinforce_group, total, lr_start, lr_end, gamma_start, gamma_end])

            # ---------------------------------------------------------- PART C
            with gr.Tab('Part C · Watch & Evaluate'):
                gr.Markdown('### Watch a trained agent land. Optionally disturb it during touchdown (C.3).')
                with gr.Row():
                    with gr.Column(scale=1):
                        watch_agent_dd = gr.Dropdown(initial_choices, value=initial_choices[0], label='Agent')
                        watch_sampled = gr.Checkbox(False, label='REINFORCE: sample actions (as in training) instead of greedy')
                        watch_seed = gr.Number(value=20000, precision=0, label='Start-condition seed (20000+ never seen in training)')
                        watch_family = gr.Dropdown(list(PERTURBATIONS), value='None', label='Perturbation')
                        watch_level = gr.Dropdown([0.0], value=0.0, label='Level (observation units)', interactive=False)
                        watch_btn = gr.Button('▶ Run & render episode', variant='primary')
                    with gr.Column(scale=2):
                        agent_video = gr.Video(label='Episode', autoplay=True, height=360)
                        agent_result = gr.Markdown()
                agent_telemetry = gr.Plot(label='Telemetry')

                gr.Markdown('---\n### C.2 Landing success on unseen start conditions')
                with gr.Row():
                    eval_agents = gr.CheckboxGroup(initial_choices, value=[c for c in initial_choices if c.startswith('Pretrained')] or initial_choices[:1], label='Agents')
                    with gr.Column():
                        eval_sampled = gr.Checkbox(False, label='REINFORCE agents: sampled actions')
                        eval_n = gr.Slider(10, 200, value=100, step=10, label='Test episodes')
                with gr.Row():
                    eval_family = gr.Dropdown(list(PERTURBATIONS), value='None', label='Perturbation')
                    eval_level = gr.Dropdown([0.0], value=0.0, label='Level', interactive=False)
                    eval_btn = gr.Button('Evaluate', variant='primary')
                eval_table = gr.Dataframe(label='Evaluation summary', interactive=False, wrap=True)
                eval_plot = gr.Plot()

                watch_family.change(update_levels, watch_family, watch_level)
                eval_family.change(update_levels, eval_family, eval_level)
                watch_btn.click(watch_agent, [watch_agent_dd, watch_sampled, watch_seed, watch_family, watch_level, session_agents],
                                [agent_video, agent_result, agent_telemetry])
                eval_btn.click(evaluate_agents, [eval_agents, eval_sampled, eval_n, eval_family, eval_level, session_agents],
                               [eval_table, eval_plot])

            with gr.Tab('Part C · Compare & Robustness'):
                gr.Markdown('### C.1 / C.2 Learning curves, sample efficiency and convergence stability')
                with gr.Row():
                    compare_agents = gr.CheckboxGroup(learned, value=learned, label='Agents with training logs')
                    compare_btn = gr.Button('Compare learning', variant='primary')
                compare_table = gr.Dataframe(label='Sample efficiency & stability', interactive=False, wrap=True)
                compare_plot = gr.Plot()

                gr.Markdown('---\n### C.3 Robustness sweep: success rate as perturbation strength grows')
                with gr.Row():
                    sweep_agents = gr.CheckboxGroup(learned, value=learned, label='Agents (greedy)')
                    sweep_family = gr.Dropdown([k for k in PERTURBATIONS if k != 'None'], value='Sideways gust at touchdown', label='Perturbation family')
                    sweep_n = gr.Slider(10, 100, value=40, step=10, label='Episodes per point')
                    sweep_btn = gr.Button('Run sweep', variant='primary')
                sweep_plot = gr.Plot()
                sweep_table = gr.Dataframe(label='Robustness table', interactive=False)

                compare_btn.click(compare_learning, [compare_agents, session_agents], [compare_table, compare_plot])
                sweep_btn.click(robustness_sweep, [sweep_agents, sweep_family, sweep_n, session_agents], [sweep_table, sweep_plot])

            with gr.Tab('Inferences'):
                gr.Markdown(INFERENCES)

        # Training wiring is after all tabs so agent lists in Part C can refresh when training finishes
        train_inputs = [algo, total, seed_train, lr_start, lr_end, gamma_start, gamma_end, eps_start, eps_end, eps_decay,
                        batch_size, target_update, buffer_size, ent_start, ent_end, eps_per_update, save_name, session_agents]
        list_outputs = [watch_agent_dd, eval_agents, compare_agents, sweep_agents]
        train_event = train_btn.click(train_agent, train_inputs, [train_status, train_plot, session_agents, weights_file],
                                      show_progress='hidden')
        train_event.then(refresh_agent_lists, session_agents, list_outputs)
        stop_btn.click(None, None, None, cancels=[train_event])
        upload_btn.click(upload_weights, [upload, upload_kind, session_agents], [upload_status, session_agents]).then(
            refresh_agent_lists, session_agents, list_outputs)

        demo.load(inspect_environment, None, [env_info, state_table, action_table, reward_table])
        demo.load(schedule_preview, schedule_inputs, schedule_plot)

    return demo


if __name__ == '__main__':
    build_app().queue().launch(theme=gr.themes.Soft(), css=CSS)
