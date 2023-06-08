"""functions to help compute the standard error in the expectation values by
using a monte-carlo based sampling approach rather than resampling from the
actual physical device
"""
# pylint: disable-msg=no-member
from typing import Dict, List, Optional

import torch

# set the random seed
torch.manual_seed(42)

from pennylane import numpy as np


def hist_to_dist(result_dict: Dict[str, float], n_qubits: Optional[int] = 5) \
                -> torch.distributions.Categorical:
    """generate a distribution object from a result histogram dict for a single
    group of Pauli words

    args:
        result_dict (Dict[str, float]): raw data dictionary containing int
            representation of bit strings as keys and empirical probabilities
            as values
        n_qubits (Optional[int]): no. of qubits in the circuit, default=5

    returns:
        (torch.distributions.Categorical): categorical distribution which can
        be used to generate more samples similar to the circuit measurement
        results
    """
    probs = []

    # iterate over all computational basis states [0, ..., 2**n_qubits] and
    # get the probability of each one (if available) to construct the
    # categorical distribution
    for outcome in range(2**n_qubits):
        key = str(outcome)
        if key in result_dict.keys():
            probs.append(result_dict[key])
        else:
            probs.append(0.)

    return torch.distributions.Categorical(torch.Tensor(probs))


def simulate_job_results(
    result_dists: List[torch.distributions.Categorical],
    shots: Optional[int] = 1000,
    seed: Optional[int] = 42
) -> List[Dict[str, Dict[str, float]]]:
    """simulate the job results for an experiment using the distribution
    objects constructed from the actual experiment results

    args:
        result_dists (List[torch.distributions.Categorical]): list of
            distribution objects constructed from the experiment results, one
            entry for each group of commuting Pauli words
        shots (Optional[int]): no. of shots to use in the MC simulation,
            default = 1000
        seed (Optional[int]): random seed for reproducibility, default = 42,
            not being set currently since this leads to a std. error of 0
    """

    # generate job results in the same format as actual HW results
    job_results = []
    for dist in result_dists:
        # sample from the distribution object corresponding to a given group
        # of commuting Pauli words
        samples = dist.sample(torch.Size([shots]))

        # generate dict with int(bitstring) as keys and normalized frequencies
        # as values
        unique_samples, unique_counts = np.unique(
            samples, return_counts=True
        )
        dist_sample_dict = {
            str(sample): counts/shots for sample, counts in zip(unique_samples,
                                                                unique_counts)
        }

        # add the dict to the results list
        job_results.append({'histogram': dist_sample_dict})

    return job_results


def compute_mean_and_std_dev(h_estimates: List[float]):
    """compute the mean and std error of multiple MC estimates of expectation
        value of the Hamiltonian

    args:
        h_estimates (List[float]): list of MC estimates of expvals

    returns:
        float, float: mean and std. error of the MC estimates
    """
    std_dev = np.std(np.array(h_estimates))
    return np.mean(h_estimates), std_dev  # /np.sqrt(len(h_estimates))
