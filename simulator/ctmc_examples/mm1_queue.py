import torch
from typing import Dict, Any

from ..ctmc_base import CTMCSimulator


class MM1Simulator(CTMCSimulator):
    """
    A specialized simulator for an M/M/1 queue.

    In this system:
      - Arrivals occur according to a Poisson process with rate λ (arrival_rate).
      - Departures occur according to a Poisson process with rate μ (service_rate).
      - There is a single server.
      - The queue length is tracked in current_states[0].

    This simulator runs several sample paths in parallel (batch simulation).
    """

    def __init__(
        self, 
        config: Dict[str, Any], 
        device: torch.device = torch.device("cpu"), 
        accuracy: int = 32,
        num_paths: int = 10, 
        seed: int = None
    ):
        """
        Initialize the M/M/1 Queue Simulator.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing 'arrival_rate', 'service_rate',
            and optionally 'num_state_variables' (should be 1 for an M/M/1 queue).
        num_paths : int, optional
            Number of sample paths to be generated in parallel, by default 10.
        device : torch.device, optional
            PyTorch device to use, by default torch.device("cpu").
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        super().__init__(config=config, device=device, accuracy=accuracy, num_paths=num_paths, seed=seed)

        # Extract arrival and service rates from config
        self.arrival_rate = config.get("arrival_rate", 1.0)
        self.service_rate = config.get("service_rate", 1.0)

        # Precompute single-step event probabilities (if you were to step in discrete time).
        # Note: These are not strictly necessary for a continuous-time simulation,
        # but can be illustrative or used for certain approximations.
        # The step method below uses the exponential distribution for the continuous case.
        self.p_arrival = 1 - torch.exp(-torch.tensor(self.arrival_rate, device=device))
        self.p_departure = 1 - torch.exp(-torch.tensor(self.service_rate, device=device))

    def _step(self) -> None:
        """
        Perform a single simulation step for the M/M/1 queue in continuous time.

        1) Compute total rate for each path: λ + μ.
        2) Sample the time increment from an Exponential distribution with parameter (λ + μ).
        3) Update each path's clock.
        4) Decide whether the event is an arrival or a departure by sampling Bernoulli(λ/(λ+μ)).
        5) Update queue lengths:
            - If arrival, current_states += 1
            - If departure AND the queue is non-empty, current_states -= 1
        """
        # 1) Calculate total rate for each sample path
        total_rate = self.arrival_rate + self.service_rate  # Scalar in this simple M/M/1 case

        # 2) Sample time increments for each path
        # Shape: (num_paths,) 
        time_increments = torch.distributions.Exponential(total_rate).sample((self.num_paths,))
        time_increments = time_increments.to(self.device)

        # 3) Update the time vector
        self.current_times += time_increments

        # 4) Determine the event: arrival (1) or departure (0)
        event_prob = self.arrival_rate / total_rate  # Probability of arrival
        events = torch.bernoulli(
            torch.full((self.num_paths,), event_prob, device=self.device)
        ).int()

        # 5) Update the queue length
        arrivals = events
        departures = 1 - events

        # For each path, departure is valid only if the queue is not empty
        # current_states shape: [S=1, num_paths] => we track queue length in row 0
        queue_lengths = self.current_states[0, :]
        non_empty = (queue_lengths > 0).int()  # 1 if queue length > 0, else 0

        # valid_departures = departure * (queue > 0)
        valid_departures = departures * non_empty

        # Update the queue length
        # For M/M/1, the queue length is stored in current_states[0,:]
        self.current_states[0, :] += arrivals - valid_departures


if __name__ == "__main__":
    # Example usage:

    # Define the configuration for the M/M/1 queue
    config = {
        "arrival_rate": 0.5,         # Average arrival rate (λ)
        "service_rate": 0.7,         # Average service rate (μ)
        "num_state_variables": 1     # For an M/M/1 queue, we typically track just 1 state variable (the queue length)
    }

    # Choose num_paths (number of parallel sample paths)
    num_paths = 5

    # Optionally specify a device (e.g. "cuda") if you have a GPU available
    device = torch.device("cpu")

    # Optionally specify a fixed seed for reproducibility
    seed = 42

    # Initialize the simulator
    simulator = MM1Simulator(config, num_paths=num_paths, device=device, seed=seed)

    # Run the simulation for 10 steps
    simulator.run_until_time(target_time=20)

    # Retrieve the simulation history
    history = simulator.history

    # Print the results
    print("Simulation History:")
    for i, record in enumerate(history, start=1):
        # record["state"] is of shape [S, num_paths] => for M/M/1, S=1
        state_vals = record["state"].squeeze(0).tolist()  # shape: [num_paths]
        time_vals = record["time"].tolist()               # shape: [num_paths]
        print(f"Step {i}:")
        print(f"  Times         : {[f'{t:.2f}' for t in time_vals]}")
        print(f"  Queue Lengths : {state_vals}")