import json
import os
import numpy as np
import jax
from copy import deepcopy

from gfn_maxent_rl.utils.exhaustive import compute_cache, push_source_flow_to_terminating_states
from gfn_maxent_rl.utils.metrics import jensen_shannon_divergence, entropy, pearson_correlation, spearman_correlation
from gfn_maxent_rl.utils.evaluations import get_samples_from_env
from gfn_maxent_rl.utils.estimation import estimate_log_probs_backward
from gfn_maxent_rl.envs.errors import StatesEnumerationError


class SyncEvaluator:
    def __init__(self, env, algorithm, path, run, target={}, n_eval=1000):
        self.env = env
        self.algorithm = algorithm
        self.run = None if ((run is None) or run.disabled) else run
        self.target = target
        self.path = path
        self.n_eval = n_eval
        self.metrics = {}
        self.step = -1

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

    def enqueue(self, params, state, step, batch_size=256):
        """Compute metrics synchronously."""
        step_metrics = {}
        
        # Skip JSD and entropy if target not available
        if 'log_probs' in self.target:
            try:
                # Only compute JSD/entropy if cache-based computation is possible
                cache = compute_cache(
                    self.env,
                    self.algorithm.log_policy,
                    params,
                    state,
                    batch_size=batch_size
                )
                
                caches = {key: np.expand_dims(log_probs, 0) for key, log_probs in cache.items()}
                mdp_state_graph = push_source_flow_to_terminating_states(self.env.mdp_state_graph, caches)
                
                distribution = dict()
                for state_node, is_terminating in mdp_state_graph.nodes(data='terminating', default=False):
                    if is_terminating:
                        log_prob = mdp_state_graph.nodes[state_node]['log_prob'][0]
                        distribution[state_node] = log_prob
                
                step_metrics.update({
                    'jsd': jensen_shannon_divergence(distribution, self.target['log_probs']),
                    'entropy': entropy(distribution),
                })
            except StatesEnumerationError:
                # Skip JSD/entropy for environments where cache cannot be computed
                pass
        
        # Compute correlation metrics using direct sampling
        try:
            # Sample terminal states from the environment
            env_copy = deepcopy(self.env)
            key = jax.random.PRNGKey(42 + step)  # Different seed for each step
            samples, returns = get_samples_from_env(
                env_copy, self.algorithm, params, state, key, 
                num_samples=self.n_eval, copy_env=False, verbose=False
            )
            
            # Compute log probabilities using backward rollout
            if len(samples) > 0:
                # Use estimate_log_probs_backward for accurate log_prob computation
                log_probs_dict = estimate_log_probs_backward(
                    env_copy, self.algorithm, params, state, samples,
                    batch_size=min(32, len(samples)), 
                    num_trajectories=32,  # Reduced for faster computation
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
            print(f"Warning: Correlation computation failed at step {step}: {e}")
        
        # Save the metrics
        if step > self.step:
            self.step = step
            for key, value in step_metrics.items():
                self.metrics[key] = value

        # Log metrics
        data = {f'metrics/{key}': value for (key, value) in step_metrics.items()}
        data['step'] = step
        log_file = os.path.join(self.path, "log.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        if self.run is not None:
            self.run.log(data)

    def join(self):
        """Return the final metrics."""
        results = dict(self.metrics)
        results['_step'] = self.step
        return results