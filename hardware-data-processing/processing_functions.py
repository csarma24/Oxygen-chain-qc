# pylint: disable-msg=too-many-locals
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


def hist_to_counts(hist, shots, reverse=True, valid_bit_strings=None):
    """Converts a histogram retrieved from IonQ results from a set of
    'backwards' bit strings and probabilities to a set of 'forwards' 
    bit strings and counts.
    
    Args:
        hist (Dict[str, float]): Raw data dictionary containing bit strings
            as keys and probabilities as values.
        num_shots (int): Number of shots the experiment was run with.
        reverse (bool): Whether to reverse the bit strings in the output. When
            using PennyLane, we want to do this so that the order matches the
            order of qubits in the Hamiltonian.
            
    Returns:
        Dict[str, int]: Dictionary of bit strings and the number of times they
        occured, based on the probability and number of shots.
    """
    if reverse:
        processed_hist = {
            np.binary_repr(int(key), 5)[::-1]: int(np.round(shots * val))
            for key, val in list(hist.items())
        }
    
        if valid_bit_strings is not None:
            return {key: val for key, val in processed_hist.items() if key in valid_bit_strings}
            
        return processed_hist
        
    
    return {
        np.binary_repr(int(key), 5): int(np.round(shots * val)) for key, val in list(hist.items())
    }


def eigenvalue(pauli, bitstring):
    """Given a Pauli word and a bitstring, return whether it is a +1 or -1 
    eigenvalue.
    
    Note that this assumes that a qubit-wise rotation back to the computational
    basis would be applied before computing the eigenvalue (essentially, every
    Pauli is considered as having only I/Z).
    
    Args:
        pauli (qml.Observable): A Pauli word
        bitstring (str): Bit string, i.e., measurement outcome in computational
            basis.
            
    Returns:
        int: The appropriate eigenvalue, +1 or -1.
    """
    eigval = 1

    pauli_string = qml.pauli.pauli_word_to_string(
        pauli, wire_map={i: i for i in range(len(bitstring))}
    )

    for bit, p in zip(bitstring, pauli_string):
        if bit == "1" and p != "I":
            eigval *= -1

    return eigval


def extract_statistics(group_paulis, group_hist):
    """Computes the expectation values and covariance matrix for 
    measurement results of a group of Paulis that were measured simultaneously.
    
    Args:
        group_paulis (List[qml.Observable]): A list of commuting
            Pauli words that were measured together.
        group_hist (Dict[str, int]): A dictionary of measurement strings
            and counts that will be postprocessed to compute the
            expectation values of observables in group_paulis.
            
    Returns:
        array(float), array(float): The expectation values for the
        set of observables, and the associated covariance matrix extracted from
        the measurement statistics.
    """
    pauli_results = []

    for pauli in group_paulis:
        eigvals_this_pauli = []
        for bitstring, count in group_hist.items():
            eigval = eigenvalue(pauli, bitstring)
            eigvals_this_pauli.extend(count * [eigval])
        pauli_results.append(eigvals_this_pauli)

    pauli_results = np.array(pauli_results)

    expvals = np.mean(pauli_results, axis=1)
    covmat = np.cov(pauli_results)

    return expvals, covmat


def compute_expal_and_std(h_tapered, job_results, shots=1000, show_plots=False):
    """Computes the expectation value and covariance matrix for
    a full Hamiltonian based on a set of grouped measurement results.

    Args:
        h_tapered (qml.Hamiltonian): The tapered Hamiltonian.
        job_results (List[Dict(str, float)): Raw results from running circuits
            to measure commuting sets of H_tapered.
        shots (int): Number of shots per experiment.

    Returns:
        float, float: The expectation value of the Hamiltonian, and the
        standard dev, computed from the provided measurement results.
    """
    # Group Hamiltonian into commuting sets in the same way as when
    # we ran on hardware.
    h_tapered_grouped = qml.Hamiltonian(h_tapered.coeffs, h_tapered.ops,
                                        grouping_type="qwc")

    # Convert raw results into nicer format
    sim_results = [
        hist_to_counts(result["histogram"], shots=shots)
        for idx, result in enumerate(job_results)
    ]

    total_expval = 0
    cumulative_var = 0

    # Loop through each group; measurement results per group are independent
    for idx, (grouping_indices, group_results) in enumerate(
        zip(h_tapered_grouped.grouping_indices, sim_results)
    ):
        group_ops = [h_tapered_grouped.ops[x] for x in grouping_indices]
        pauli_labels = [qml.pauli.pauli_word_to_string(
            pauli, wire_map={i: i for i in range(5)}
        ) for pauli in group_ops]
        group_coeffs = [h_tapered_grouped.coeffs[x] for x in grouping_indices]

        expvals, covmat = extract_statistics(group_ops, group_results)

        total_expval += np.dot(group_coeffs, expvals)

        # Compute the total variance

        # First variances from the individual terms
        total_var = np.sum(
            [covmat[i, i] * (group_coeffs[i] ** 2) for i in range(len(
                group_coeffs
            ))]
        )

        # Next we add the covariance terms
        for i in range(covmat.shape[0] - 1):
            for j in range(i + 1, covmat.shape[0]):
                total_var += 2 * group_coeffs[i] * group_coeffs[j] \
                    * covmat[i, j]

        cumulative_var += total_var

        if show_plots:
            plt.matshow(covmat, vmin=-1, vmax=1)
            plt.xticks(range(len(pauli_labels)), labels=pauli_labels,
                       rotation='vertical')
            plt.yticks(range(len(pauli_labels)), labels=pauli_labels)
            plt.colorbar()
            plt.show()
            #plt.savefig(f"group{idx}.pdf")

    std = np.sqrt(cumulative_var)

    return total_expval, std
