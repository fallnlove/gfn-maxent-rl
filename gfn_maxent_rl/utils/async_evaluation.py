import json
import os
import numpy as np
import multiprocessing as mp
import jax
from copy import deepcopy

from queue import Empty as EmptyException
from collections import defaultdict

from gfn_maxent_rl.utils.exhaustive import compute_cache, push_source_flow_to_terminating_states
from gfn_maxent_rl.utils.metrics import jensen_shannon_divergence, entropy, pearson_correlation, spearman_correlation
from gfn_maxent_rl.utils.evaluations import get_samples_from_env
from gfn_maxent_rl.utils.estimation import estimate_log_probs_backward
from gfn_maxent_rl.envs.errors import StatesEnumerationError


class AsyncEvaluator:
    def __init__(self, env, algorithm, path, run, ctx=None, target={}, n_eval=1000):
        self.env = env
        self.algorithm = algorithm
        self.run = None if ((run is None) or run.disabled) else run
        self.ctx = mp.get_context(ctx)
        self.target = target
        self.path = path
        self.n_eval = n_eval

        self._log_policy = jax.jit(algorithm.log_policy)

        self._manager = self.ctx.Manager()
        self._queue = self._manager.Queue()
        self._namespace = self._manager.Namespace()
        self._namespace.step = -1
        self._namespace.metrics = self._manager.dict()
        self._process = self.ctx.Process(
            target=AsyncEvaluator._compute_metrics,
            args=(self._queue, self._namespace, env, algorithm, target, self.run, self.path, self.n_eval),
            daemon=True
        )
        self._process.start()

    def enqueue(self, params, state, step, batch_size=256):
        # For environments where cache cannot be computed, 
        # we pass params and state directly for correlation computation
        self._queue.put((step, None, params, state))

    def join(self):
        self._queue.put(None)
        self._process.join()

        results = dict(self._namespace.metrics)
        results['_step'] = self._namespace.step
        return results

    @staticmethod
    def _compute_metrics(queue, namespace, env, algorithm, target, run, path, n_eval):
        terminate = False
        while not terminate:
            # Create the batch of data
            steps, params_list, states_list = [], [], []
            while True:
                try:
                    result = queue.get(block=True, timeout=1)

                    if result is None:
                        terminate = True
                        break

                    step, cache, params, state = result
                    steps.append(step)
                    params_list.append(params)
                    states_list.append(state)
                except EmptyException:
                    break

            # Process the batch
            if steps:
                # Compute the metrics
                metrics = dict()
                for i, (step, params, state) in enumerate(zip(steps, params_list, states_list)):
                    step_metrics = {}
                    
                    # Skip JSD and entropy if target not available
                    if 'log_probs' in target:
                        try:
                            # Only compute JSD/entropy if cache-based computation is possible
                            cache = compute_cache(
                                env,
                                algorithm.log_policy,
                                params,
                                state,
                                batch_size=256
                            )
                            
                            caches = {key: np.expand_dims(log_probs, 0) for key, log_probs in cache.items()}
                            mdp_state_graph = push_source_flow_to_terminating_states(env.mdp_state_graph, caches)
                            
                            distribution = dict()
                            for state_node, is_terminating in mdp_state_graph.nodes(data='terminating', default=False):
                                if is_terminating:
                                    log_prob = mdp_state_graph.nodes[state_node]['log_prob'][0]
                                    distribution[state_node] = log_prob
                            
                            step_metrics.update({
                                'jsd': jensen_shannon_divergence(distribution, target['log_probs']),
                                'entropy': entropy(distribution),
                            })
                        except StatesEnumerationError:
                            # Skip JSD/entropy for environments where cache cannot be computed
                            pass
                    
                    # Compute correlation metrics using direct sampling
                    try:
                        # Sample terminal states from the environment
                        env_copy = deepcopy(env)
                        key = jax.random.PRNGKey(42 + step)  # Different seed for each step
                        samples, returns = get_samples_from_env(
                            env_copy, algorithm, params, state, key, 
                            num_samples=n_eval, copy_env=False, verbose=False
                        )
                        
                        # Compute log probabilities using backward rollout
                        if len(samples) > 0:
                            # Use estimate_log_probs_backward for accurate log_prob computation
                            log_probs_dict = estimate_log_probs_backward(
                                env_copy, algorithm, params, state, samples,
                                batch_size=min(32, len(samples)), 
                                num_trajectories=100,
                                verbose=False
                            )
                            
                            # Extract log_probs in the same order as samples
                            sampled_log_probs = np.array([log_probs_dict[sample] for sample in samples])
                        else:
                            sampled_log_probs = np.array([])
                        
                        # Compute correlations
                        if len(sampled_log_probs) > 1 and len(returns) > 1:
                            pearson_corr, pearson_p = pearson_correlation(sampled_log_probs, returns)
                            spearman_corr, spearman_p = spearman_correlation(sampled_log_probs, returns)
                            
                            step_metrics.update({
                                'pearson_correlation': pearson_corr,
                                'pearson_p_value': pearson_p,
                                'spearman_correlation': spearman_corr,
                                'spearman_p_value': spearman_p,
                                'n_samples_correlation': len(sampled_log_probs)
                            })
                        else:
                            step_metrics.update({
                                'pearson_correlation': np.nan,
                                'pearson_p_value': np.nan,
                                'spearman_correlation': np.nan,
                                'spearman_p_value': np.nan,
                                'n_samples_correlation': len(sampled_log_probs) if len(sampled_log_probs) > 0 else 0
                            })
                            
                    except Exception as e:
                        # If correlation computation fails, set NaN values
                        step_metrics.update({
                            'pearson_correlation': np.nan,
                            'pearson_p_value': np.nan,
                            'spearman_correlation': np.nan,
                            'spearman_p_value': np.nan,
                            'n_samples_correlation': 0
                        })
                    
                    metrics[step] = step_metrics

                for step, metric in metrics.items():
                    # Save the metrics of the latest step
                    if step > namespace.step:
                        namespace.step = step
                        for key, value in metric.items():
                            namespace.metrics[key] = value

                    # Send to Wandb
                    data = {f'metrics/{key}': value for (key, value) in metric.items()}
                    data['step'] = step
                    log_file = os.path.join(path, "log.jsonl")
                    with open(log_file, "a", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                        f.write("\n")
                    if run is not None:
                        run.log(data)
