import json
import os
import time
import jax.numpy as jnp
import numpy as np
import jax
import optax
import hydra
import networkx as nx

from numpy.random import default_rng
from tqdm.auto import trange
from tqdm import tqdm

from gfn_maxent_rl.utils.exhaustive import exact_log_posterior
from gfn_maxent_rl.utils.sync_evaluation import SyncEvaluator
from gfn_maxent_rl.envs.errors import StatesEnumerationError
from gfn_maxent_rl.utils.evaluations import sample_trajectories_for_training


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
    # Set the RNGs for reproducibility
    rng = default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Create the environment
    # Train environment
    env, infos = hydra.utils.instantiate(
        config.env,
        num_envs=config.num_envs,
        seed=config.seed,
        rng=rng,
    )
    # Evaluation environment
    env_valid, _ = hydra.utils.instantiate(
        config.env,
        num_envs=config.num_envs,
        seed=config.seed,
        rng=rng,
    )

    if 'graph' in infos:
        ground_truth = nx.to_numpy_array(infos['graph'], weight=None)

    # Add wrapper to the environment
    if config.reward_correction:
        env = hydra.utils.instantiate(config.env_wrapper, env=env)
        env_valid = hydra.utils.instantiate(config.env_wrapper, env=env_valid)

    # Create the algorithm
    algorithm = hydra.utils.instantiate(config.algorithm, env=env)
    # Create transition steps schedule
    # assert config.num_iterations == config.lr1.decay_steps + config.lr1.warmup_steps
    # assert config.num_iterations == config.lr2.decay_steps + config.lr2.warmup_steps
    algorithm.optimizer = hydra.utils.instantiate(config.optimizer)
    params, state = algorithm.init(key)

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - config.exploration.min_exploration),
        transition_steps=config.exploration.warmup,
    ))

    target = {}
    try:
        target['log_probs'] = exact_log_posterior(env, batch_size=config.batch_size)
    except StatesEnumerationError:
        pass
    path = f"tmp_time/{config.env.dataset_name}/run_{config.seed}"
    os.makedirs(path, exist_ok=True)
    warmup_iterations = 10

    for iteration in tqdm(range(warmup_iterations)):
        epsilon = exploration_schedule(iteration)
        # Sample actions from the model (with exploration)
        samples, key = sample_trajectories_for_training(env, algorithm, params, state.network, key, epsilon)
        params, state, logs = algorithm.step(params, state, samples)

    start_time = time.perf_counter()

    for iteration in tqdm(range(warmup_iterations, config.num_iterations + warmup_iterations)):
        epsilon = exploration_schedule(iteration)
        # Sample actions from the model (with exploration)
        samples, key = sample_trajectories_for_training(env, algorithm, params, state.network, key, epsilon)
        params, state, logs = algorithm.step(params, state, samples)

    end_time = time.perf_counter()

    with open(os.path.join(path, "log.jsonl"), "w") as f:
        json.dump({'Training time': end_time - start_time}, f)

if __name__ == '__main__':
    main()
