import torch
from typing import Dict, Any, List, Optional

from .ctmc_base import CTMCSimulator


class CTMDPSimulator(CTMCSimulator):
    """
    A general Continuous-Time Markov Decision Process (CTMDP) simulator.

    In addition to the usual CTMC dynamics, a CTMDP allows for actions
    that can affect the transition rates and produce rewards.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        accuracy: int = 32,
        num_paths: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the CTMDP Simulator.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing system parameters such as:
                - 'num_state_variables': number of dimensions in the state
                - possibly other fields needed for rate computations
        num_paths : int, optional
            Number of sample paths to be generated in parallel, by default 10.
        device : torch.device, optional
            PyTorch device to use (e.g., 'cpu' or 'cuda'), by default torch.device("cpu").
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        super().__init__(config=config, device=device, accuracy=accuracy, num_paths=num_paths, seed=seed)
        self.A = self.config.get("num_action_variables", 1)

        # Store the action taken at the most recent step for each sample path.
        self.next_actions = torch.zeros(
            (self.A, self.num_paths), 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )
        # Store the reward obtained at the most recent step for each sample path.
        self.next_reward = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )
        # Store the total reward obtained at the most recent step for each sample path.
        self.total_reward = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )

    def _update_actions(self) -> torch.Tensor:
        """
        Compute or retrieve the last action for each sample path, given the current state.
        Updates self.next_actions with shape (A, num_paths). 
        """
        # This is a placeholder for the actual implementation of the policy.
        pass

    def _update_rewards(self):
        """
        Compute the reward for each sample path, given the current state and action.
        Updates self.next_reward with shape (num_paths). 
        Updates self.total_reward with shape (num_paths). 
        """
        # This is a placeholder for the actual implementation of the reward function.
        pass

    def _step(self) -> None:
        """
        Perform a single simulation step for the CTMDP:
        
        1. Get the current action for each sample path.
        2. Compute the immediate reward.
        3. Compute transition rates as a function of (state, action).
        4. Sample the time increment from an exponential with parameter = sum of rates.
        5. Sample which event occurs for each path (categorical distribution).
        6. Update the state based on the chosen event.
        7. Store the reward for future summarization.
        """
        self._update_actions()
        self._update_transition_rates()
        self._update_next_events()
        self._update_times_and_steps() 
        self._update_rewards()
        self._update_states()

    def _summarize(self) -> None:
        """
        Summarize the current batch of states, times, and rewards, 
        and record them in the history.
        """
        # Save state, time, and last reward
        self.history.append({
            "state": self.current_states.clone().cpu(),
            "time": self.current_times.clone().cpu(),
            "reward": self.next_reward.clone().cpu()
        })

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
                original_total_reward = self.total_reward.clone()

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
                self.total_reward = torch.where(
                    active_mask, 
                    self.total_reward, 
                    original_total_reward
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
                original_total_reward = self.total_reward.clone()
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
                self.total_reward = torch.where(
                    active_mask, 
                    self.total_reward, 
                    original_total_reward
                )

                self._summarize()

                # Update the mask for active paths
                active_mask = self.current_times < target_time


if __name__ == "__main__":
    ##########################################
    # Example usage of CTMDPSimulator
    ##########################################
    
    config = {
        "num_state_variables": 1
        # Add other config fields as needed for your transitions or rates
    }

    simulator = CTMDPSimulator(
        config=config,
        num_paths=5, 
        device=torch.device("cpu"), 
        seed=42
    )

    # Run for 5 steps
    simulator.run(num_steps=5)

    # Print history
    for i, record in enumerate(simulator.history, 1):
        print(f"Step {i}")
        print("  Time   :", record["time"].tolist())
        print("  State  :", record["state"].squeeze(0).tolist())  # Since S=1
        print("  Reward :", record["reward"].tolist())
        print()
