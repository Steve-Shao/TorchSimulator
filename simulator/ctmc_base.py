import torch
from typing import Dict, Any, List


class CTMCSimulator:
    """
    A base class for Continuous-Time Markov Chain (CTMC) simulators.
    It handles common simulation constructs like:
      - state vectors,
      - time vectors,
      - history recording,
      - batch size management.

    Subclasses should override the `step` method to implement specific CTMC dynamics.
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
        Initialize the CTMC Simulator.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing system parameters such as rates,
            state dimensions, etc.
        num_paths : int, optional
            Number of sample paths to be generated in parallel, by default 10.
        device : torch.device, optional
            PyTorch device to use (e.g., 'cpu' or 'cuda'), by default torch.device("cpu").
        seed : int, optional
            Random seed for reproducibility, by default None (no fixed seed).
        """
        # ========== Store the configuration and device ========== # 
        self.config = config
        self.device = device
        self.dtype_int = getattr(torch, f'int{accuracy}')
        self.dtype_float = getattr(torch, f'float{accuracy}')
        self.num_paths = num_paths

        # ========== Set random seed (optional) ========== # 
        if seed is not None:
            torch.manual_seed(seed)
        
        # ========== Initialize the system ========== # 
        # Determine the number of dimensions (S) for the state (e.g., queue length, etc.)
        self.S = self.config.get("num_state_variables", 1)
        self.E = self.config.get("num_event_types", 1)
        # State vector shape: [S, num_paths]
        self.current_states = torch.zeros(
            (self.S, self.num_paths), 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )
        # Time vector shape: [num_paths], one time value per sample path
        self.current_times = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )
        # Step vector shape: [num_paths], one step value per sample path
        self.current_steps = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )
        # ========== Initialize placeholder variables ========== # 
        # Rate vector shape: [A, num_paths], one rate value per action type per sample path
        self.next_rates = torch.ones(
            (self.E, self.num_paths), 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )
        # Total rate vector shape: [num_paths], one rate value per sample path
        self.next_rates_total = torch.sum(self.next_rates, dim=0)
        # Time increment vector shape: [num_paths], one time increment per sample path
        self.next_time_increments = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )
        # Event index vector shape: [num_paths], one event index per sample path
        self.next_event_indices = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )

        # ========== Initialize list to record the simulation history ========== #  =
        self.history: List[Dict[str, torch.Tensor]] = []

    def _update_transition_rates(self):
        """
        Compute the transition rates for each possible event, given the current state and the chosen action.
        Updates self.next_rates with shape (E, num_paths). 
        Updates self.next_rates_total with shape (num_paths). 
        """
        # This is a placeholder for the actual implementation of the transition rates.
        pass

        # Note that we need to ensure all next_rates_total are strictly positive
        if not torch.all(self.next_rates_total > 0):
            raise ValueError("All next_rates_total must be strictly positive.")
    
    def _update_next_events(self):
        """
        Prepare the next event for each sample path.
        Updates self.next_event_indices with shape (num_paths). 
        Updates self.next_time_increments with shape (num_paths). 
        """
        # Sample exponential random variables for each event type and path
        # Shape: [E, num_paths]
        exp_samples = torch.empty(
            (self.E, self.num_paths),
            dtype=self.dtype_float,
            device=self.device
        ).exponential_() / self.next_rates

        # Find the minimum time increment for each path
        # Shape: [num_paths]
        self.next_time_increments, self.next_event_indices = exp_samples.min(dim=0)

    def _update_times_and_steps(self):
        """
        Update the time vector for each sample path.
        Updates self.current_times with shape (num_paths). 
        Updates self.current_steps with shape (num_paths). 
        """
        self.current_times += self.next_time_increments
        self.current_steps += 1

    def _update_states(self):
        """
        Update the state based on which event occurred for each sample path.
        Updates self.current_states with shape (S, num_paths). 
        """
        # This is a placeholder for the actual implementation of the state update.
        pass

    def _step(self) -> None:
        """
        Perform a single simulation step.
        """
        self._update_transition_rates()
        self._update_next_events()
        self._update_times_and_steps()  
        self._update_states()

    def _summarize(self) -> None:
        """
        Summarize the current batch of states and record them into the history.

        This method creates a dictionary containing the current state and time vectors,
        and appends it to the history list. Subclasses can override to record additional
        statistics (e.g., performance metrics).
        """
        # Clone the current state and time tensors to avoid mutations later
        self.history.append({
            "state": self.current_states.clone().cpu(),
            "time": self.current_times.clone().cpu()
        })

    def run(self, num_steps: int) -> None:
        """
        Run the simulation for a specified number of steps.

        Parameters
        ----------
        num_steps : int
            Number of simulation steps to perform.
        """
        with torch.no_grad():
            for _ in range(num_steps):
                self._step()
                self._summarize()
                self.current_steps += 1

    def run_until_step(self, target_step: int) -> None:
        """
        Run the simulation until the current step reaches the target_step.

        Parameters
        ----------
        target_step : int
            The step number at which to stop the simulation.
        """
        with torch.no_grad():
            active_mask = self.current_steps < target_step
            while torch.any(active_mask):

                # Update only active paths by masking
                original_state = self.current_states.clone()
                original_time = self.current_times.clone()
                original_step = self.current_steps.clone()

                # Perform a step
                self._step()

                # Apply mask: keep updates only where active
                self.current_states = torch.where(
                    active_mask.unsqueeze(0), 
                    self.current_states, 
                    original_state
                )
                self.current_times = torch.where(
                    active_mask, 
                    self.current_times, 
                    original_time
                )
                self.current_steps = torch.where(
                    active_mask, 
                    self.current_steps + 1, 
                    original_step
                )

                self._summarize()

                # Update the mask for active paths
                active_mask = self.current_steps < target_step

    def run_until_time(self, target_time: float) -> None:
        """
        Run the simulation until all sample paths exceed the target_time.

        Parameters
        ----------
        target_time : float
            The time threshold to stop the simulation.
        """
        with torch.no_grad():
            active_mask = self.current_times < target_time
            while torch.any(active_mask):

                # Update only active paths by masking
                original_state = self.current_states.clone()
                original_time = self.current_times.clone()
                original_step = self.current_steps.clone()

                # Perform a step
                self._step()

                # Apply mask: keep updates only where active
                self.current_states = torch.where(
                    active_mask.unsqueeze(0), 
                    self.current_states, 
                    original_state
                )
                self.current_times = torch.where(
                    active_mask, 
                    self.current_times, 
                    original_time
                )
                self.current_steps = torch.where(
                    active_mask, 
                    self.current_steps + 1, 
                    original_step
                )

                self._summarize()

                # Update the mask for active paths
                active_mask = self.current_times < target_time


if __name__ == "__main__":
    ##########################################
    # Comprehensive Testing of CTMCSimulator
    ##########################################
    
    import matplotlib.pyplot as plt

    # Configuration for the simulator
    config = {
        "num_state_variables": 2  # Example with 2 state variables
    }

    # Initialize the simulator with dummy parameters
    simulator = CTMCSimulator(
        config=config,
        num_paths=3,
        device=torch.device("cpu"),
        seed=42
    )

    # Run the simulation for 5 steps
    simulator.run(num_steps=5)

    # Print the simulation history
    print("Simulation History:")
    for step, record in enumerate(simulator.history):
        state = record['state'].numpy()
        time = record['time'].numpy()
        print(f"Step {step + 1}:")
        print(f"  State:\n{state}")
        print(f"  Time: {time}\n")
