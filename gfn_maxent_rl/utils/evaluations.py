import numpy as np
import jax.numpy as jnp

from copy import deepcopy
from tqdm.auto import trange


def get_samples_from_env(
        env,
        algorithm,
        params,
        net_state,
        key,
        num_samples=1000,
        copy_env=True,
        verbose=False,
        **kwargs
    ):
    """Get samples by running the policy in the environment.

    Parameters
    ----------
    env : gym.vector.VectorEnv instance
        The environment.

    algorithm : BaseAlgorithm instance
        The algorithm. This must implement the `log_policy` method.

    params : Any
        The parameters of the networks (e.g., the policy network).
        Note that this must be the parameters of the *online* network
        (i.e., the parameters learned by the algorithm).

    net_state : Any
        The state of the network. This will be typically `state.network`,
        where `state` is the state returned by the initialization of the
        algorithm (and updated during training).

    key : jax.random.PRNGKey
        The Jax random key.

    num_samples : int
        The number of samples to return.

    copy_env : bool
        Whether `env` must be (deep)copied or not. This is useful if `env`
        is the instance of the environment used for training, to avoid any
        interaction. Set to False if we have an explicit validation env.

    verbose : bool
        Display a progress bar.

    Returns
    -------
    samples : list of samples
        A list of samples, of length `num_samples`. The samples are hashable
        keys, dependent on the environment (e.g., a tuple for Treesample envs).

    returns : np.ndarray, shape `(num_samples,)`
        The return of the trajectory for each sample. If the environment is not
        wrapped with a reward correction, then this corresponds exactly to
        the log-reward of each sample.
    """
    samples, returns = [], []
    if copy_env:
        env = deepcopy(env)
    observations, _ = env.reset()

    returns_ = np.zeros((env.num_envs,), dtype=np.float64)
    with trange(num_samples, disable=(not verbose), **kwargs) as pbar:
        while len(samples) < num_samples:
            keys = env.observation_to_key(observations)

            # Sample actions from the model (w/o exploration)
            actions, key, _ = algorithm.act(
                params, net_state, key, observations, epsilon=1.)
            actions = np.asarray(actions)

            # Apply the actions in the environment
            observations, rewards, dones, *_ = env.step(actions)

            # Compute the returns
            returns_ = returns_ + rewards * (1. - dones)

            # Add samples from the complete trajectories
            samples.extend([key for (key, done) in zip(keys, dones) if done])
            returns.extend([return_ for (return_, done) in zip(returns_, dones) if done])
            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))

            # Reset the returns for complete trajectories
            returns_[dones] = 0.

    samples = samples[:num_samples]
    returns = returns[:num_samples]

    return (samples, np.asarray(returns))


def sample_trajectories_for_training(
    env,
    algorithm,
    params,
    net_state,
    key,
    epsilon=0.1,
    verbose=False,
    **kwargs
):
    """Sample complete trajectories with exploration for training.

    Parameters
    ----------
    env : gym.vector.VectorEnv instance
        The environment.

    algorithm : BaseAlgorithm instance
        The algorithm. This must implement the `act` method.

    params : Any
        The parameters of the networks (e.g., the policy network).
        Note that this must be the parameters of the *online* network
        (i.e., the parameters learned by the algorithm).

    net_state : Any
        The state of the network. This will be typically `state.network`,
        where `state` is the state returned by the initialization of the
        algorithm (and updated during training).

    key : jax.random.PRNGKey
        The Jax random key.

    epsilon : float
        The exploration parameter. Higher values mean more exploration.

    verbose : bool
        Display a progress bar.

    Returns
    -------
    trajectories : list of trajectories
        A list of complete trajectories (one per environment), each trajectory is a dict containing:
        - 'observations': list of observations
        - 'actions': list of actions
        - 'rewards': list of rewards
        - 'dones': list of done flags
        - 'sample': the final sample key
        - 'return': the total return of the trajectory

    key : jax.random.PRNGKey
        Updated random key.
    """
    trajectories = {
        'observation': {
            'sequences': [],
            'type': [],
            'tree': [],
            'mask': [],
        },
        'next_observation': {
            'sequences': [],
            'type': [],
            'tree': [],
            'mask': [],
        },
        'action': [],
        'reward': [],
        'done': [],
    }
    
    # Reset environment for all environments
    observations, _ = env.reset()
    
    # Run trajectories until all environments are done
    for i in range(env.max_length - 1):
        # Store current observations
        for obs_key in trajectories['observation'].keys():
            trajectories['observation'][obs_key].append(observations[obs_key])
        
        # Sample actions from the model with exploration
        actions, key, _ = algorithm.act(
            params.online, net_state, key, observations, epsilon=epsilon)
        actions = np.asarray(actions)
        
        # Store actions
        trajectories['action'].append(actions)
        
        # Apply the actions in the environment
        next_observations, rewards, dones, *_ = env.step(actions)
        
        # Store rewards and dones
        trajectories['reward'].append(rewards)
        trajectories['done'].append(dones)
        # trajectories['next_observation'].append(next_observations)
        for obs_key in trajectories['next_observation'].keys():
            trajectories['next_observation'][obs_key].append(next_observations[obs_key])

        
        observations = next_observations

    trajectories['reward'] = np.concatenate(trajectories['reward'], axis=0).reshape(-1, 1)
    for obs_key in trajectories['next_observation'].keys():
        trajectories['next_observation'][obs_key] = np.concatenate(trajectories['next_observation'][obs_key], axis=0)
        trajectories['observation'][obs_key] = np.concatenate(trajectories['observation'][obs_key], axis=0)
    trajectories['action'] = jnp.concatenate(trajectories['action'], axis=0).reshape(-1, 1)
    trajectories['done'] = np.concatenate(trajectories['done'], axis=0).reshape(-1, 1)
    
    return trajectories, key
