import torch
import pandas as pd
from typing import Dict, Any, Optional

from ..ctmdp_examples.call_center import CallCenterSimulator


class CallCenterNNSimulator(CallCenterSimulator):
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        accuracy: int = 32,
        num_paths: int = 10,
        seed: Optional[int] = None
    ):
        super().__init__(config, device, accuracy, num_paths, seed)
    
    def _update_control_rules(self):
        """
        Update the control rules for the call center system.
        In the current implementation, we use constant, pre-emptive, priority-based control rules.
        Each sample path has its own priority order based on the policy rule.
        """
        self.mu_theta_diff = torch.sub(self.mu_hourly, self.theta_hourly)
        self._zero_float = torch.zeros(1, device=self.device, dtype=self.dtype_float)
        self.c_mu_theta = torch.where(
            self.theta_hourly != 0,
            self.cost_total_hourly * self.mu_hourly / self.theta_hourly,
            self._zero_float
        )
        self.c_mu = torch.mul(self.cost_total_hourly, self.mu_hourly)
        self.c_mu_theta_diff = torch.mul(self.cost_total_hourly, self.mu_theta_diff)
        
        self.policy = self.config.get("policy", "cost")
        if self.policy == "cost":
            self.kappa = -1.0 * self.cost_total_hourly
        elif self.policy == "c_mu_theta":
            self.kappa = -1.0 * self.c_mu_theta
        elif self.policy == "c_mu":
            self.kappa = -1.0 * self.c_mu
        elif self.policy == "c_mu_theta_diff":
            self.kappa = -1.0 * self.c_mu_theta_diff
        elif self.policy == "mu_theta_diff":
            self.kappa = -1.0 * self.mu_theta_diff
        else:
            raise ValueError(f"Invalid policy: {self.policy}")
        
        # Expand priority_order for each sample path
        sorted_indices = torch.argsort(self.kappa)
        self.priority_order = sorted_indices.unsqueeze(0).repeat(self.num_paths, 1)

    def _update_actions(self):
        """
        Compute or retrieve the current action for each sample path, given the current state.
        This implements a priority-based queueing discipline where:
        1. Customer classes are ordered by priority based on the policy rule (cost, c_mu_theta, etc.)
        2. Available agents are assigned to customers in order of class priority
        3. Within each class, agents serve as many customers as possible up to:
           - The number of waiting customers in that class
           - The number of remaining available agents
        4. Any remaining agents move on to serve the next priority class
        Updates self.next_actions with shape (A, num_paths). 
        Updates self.waiting_customers with shape (S, num_paths). 
        """
        # Update the current intervals to locate data from the 5-minute intervals
        self.current_intervals = torch.floor(self.current_times * self.hour_to_interval_scaler).to(self.dtype_int)
        # Cap current intervals at max_interval
        self.current_intervals = torch.minimum(
            self.current_intervals,
            torch.tensor(self.max_interval, dtype=self.dtype_int, device=self.device)
        )

        # Reset the last actions to 0
        self.next_actions.zero_()
        
        # Keep track of remaining agents for each path
        remaining_agents = self.num_server[self.current_intervals].clone()
        
        # Iterate over priority levels
        for priority in range(self.S):
            # Get class indices for current priority level across all paths
            class_idxs = self.priority_order[:, priority]
            
            # Gather the number of waiting customers for the selected classes
            waiting = self.current_states[class_idxs, torch.arange(self.num_paths, device=self.device)]
            
            # Determine how many callers can be served
            callers_served = torch.minimum(waiting, remaining_agents)
            
            # Update actions
            self.next_actions[class_idxs, torch.arange(self.num_paths, device=self.device)] = callers_served
            
            # Update remaining agents
            remaining_agents -= callers_served
        
        # Update the number of waiting customers
        self.waiting_customers = self.current_states - self.next_actions 


if __name__ == "__main__":

    ########################################################
    # Example Usage
    ########################################################
    # Load system data from CSV files
    dim = 17
    data_dir = f"configs/call_center/config_{dim}dim/"

    lambd_5min = pd.read_csv(data_dir + f"main_test_total_arrivals_partial_5min.csv", header=None)[0].to_numpy()  # Arrival rates
    mu_hourly = pd.read_csv(data_dir + f"mu_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Service rates
    theta_hourly = pd.read_csv(data_dir + f"theta_hourly_{dim}dim.csv", header=None)[0].to_numpy()  # Abandonment rates 
    arr_cdf = pd.read_csv(data_dir + f"cdf_{dim}dim.csv", header=None, delimiter=",").to_numpy() 
    cost_holding_hourly = pd.read_csv(data_dir + f"hourly_holding_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    cost_abandonment = pd.read_csv(data_dir + f"abandonment_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    cost_total_hourly = pd.read_csv(data_dir + f"hourly_total_cost_{dim}dim.csv", header=None)[0].to_numpy() 
    num_server = pd.read_csv(data_dir + f"main_test_agents.csv", header=None)[0].to_numpy()  # Arrival rates
    num_server_init = pd.read_csv(data_dir + f"initialization_{dim}dim.csv", header=None)[0].to_numpy() 

    # Model configuration dictionary
    config = {
        "_comment": "Dynamic scheduling config for call center system",
        "num_state_variables": dim,
        "policy": "c_mu_theta",
        "num_interval": 204,
        "lambda_5min": lambd_5min,
        "mu_hourly": mu_hourly, 
        "theta_hourly": theta_hourly,
        "arr_cdf": arr_cdf,
        "cost_holding_hourly": cost_holding_hourly,
        "cost_abandonment": cost_abandonment,
        "cost_total_hourly": cost_total_hourly,
        "num_server": num_server,
        "num_server_init": num_server_init
    }

    # Set device
    # device = "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    import time
    from datetime import datetime

    print(f"Starting simulation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize the simulator
    simulator = CallCenterNNSimulator(
        config=config,
        num_paths=10000,
        device=device,
        seed=42
    )

    simulator.run_until_time(target_time=0.1)

    # # Print all history records
    # print("=== Full History ===")
    # for i, record in enumerate(simulator.history):
    #     print(f"\nRecord {i}:")
    #     print("Time  :", record["time"].tolist())
    #     print("State :", record["state"].t().tolist())
    #     print("Reward:", record["reward"].tolist())

    # Print final state and time
    print("\n=== Final Results ===")
    print("Terminal State:", [f"{x:.2f}" for x in simulator.current_states.float().mean(dim=1).tolist()])
    print("Terminal Time:", f"{simulator.current_times.mean().item():.2f}")
    
    print("Total Loss:", f"{simulator.total_reward.mean().item():.2f}")  # Negative since rewards are costs
    print("Total Loss by Class:", [f"{x:.2f}" for x in simulator.total_reward_by_class.mean(dim=1).tolist()])
    
    print(f"Ending simulation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")