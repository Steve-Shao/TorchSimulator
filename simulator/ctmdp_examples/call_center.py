import torch
import pandas as pd
from typing import Dict, Any, List, Optional, Callable

from ..ctmdp_base import CTMDPSimulator  


class CallCenterSimulator(CTMDPSimulator):
    """
    A CTMDP simulator for a multi-class call center problem with:
      - S classes of callers,
      - Time-varying arrival rates (lambda_s(t)),
      - Service rates (mu_s),
      - Abandonment rates (theta_s),
      - Cost per waiting caller c_s,
      - A finite planning horizon [0, T],
      - A single pool of homogeneous agents, N(t),
      - A terminal (overtime) cost g(x) = bar_c * (1^T x - N(T))^+.

    The state is X(t) = (X_1(t), ..., X_S(t)), where X_s(t) is the number of
    waiting callers of class s at time t.

    The action is psi(t) = (psi_1(t), ..., psi_K(t)), where psi_k(t) is the number
    of callers of class s being served at time t. The constraints are:
      psi_s(t) <= X_s(t)
      1^T psi(t) = min(1^T X(t), N(t))

    The (instantaneous) cost at time t is 
        c^top [X(t) - psi(t)], 
    i.e. the sum of c_s * (X_s(t) - psi_s(t)) over s.

    There are 3S event types (arrival, service completion, abandonment for each class).
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
        Initialize the call center CTMDP simulator.

        Parameters
        ----------
        config : dict
            - "S" : int
                The number of classes.
            - "T" : float
                The time horizon.
            - "lambda_5min" : List[float]
                A list of length S for the arrival rates at 5-minute intervals.
            - "mu_hourly" : List[float]
                A list of length S for service rates (mu_s).
            - "theta_hourly" : List[float]
                A list of length S for abandonment rates (theta_s).
            - "arr_cdf" : List[float]
                A list of length S for the cumulative distribution function (CDF) of the conditional probability of arrival going to each class.
            - "cost_holding_hourly" : List[float]
                A list of length S for the holding cost rates (h_s).
            - "cost_abandonment" : List[float]
                A list of length S for the abandonment cost rates (p_s).
            - "cost_total_hourly" : List[float]
                A list of length S for the total cost rates (h_s + theta_s * p_s).
            - "num_server" : List[float]
                A list of length S for the number of available agents at time t.
            - "bar_c" : float
                The overtime cost coefficient for the terminal cost function.
            - "num_state_variables": int
                Should be = S (the dimension of X(t)).
        num_paths : int, optional
            Number of sample paths run in parallel, by default 10.
        device : torch.device, optional
            Device to use (CPU or GPU), by default torch.device("cpu").
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        # Call parent constructor first
        super().__init__(config=config, device=device, accuracy=accuracy, num_paths=num_paths, seed=seed)

        # Explicitly set dtypes based on accuracy parameter
        self.dtype_float = torch.float32 if accuracy == 32 else torch.float64
        self.dtype_int = torch.int32  # Using int32 since we don't need int64 for this simulation
        
        # ========== Overwrite related variables ========== #
        self.S = self.config.get("num_state_variables", 1)
        self.E = 3
        self.A = self.S
        
        self.current_states = torch.zeros(
            (self.S, self.num_paths), 
            dtype=self.dtype_int,  # Using int32 for discrete states
            device=self.device,
            requires_grad=False
        )        
        self.waiting_customers = torch.zeros(
            (self.S, self.num_paths), 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )
        self.next_rates = torch.ones(
            (self.E, self.num_paths), 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )
        self.next_actions = torch.zeros(
            (self.A, self.num_paths), 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )

        # ========== Additional placeholder variables ========== #
        # The unit of time in this system is hour. 
        # However, for some reason, the arrival rate is given in 5-minute intervals.
        self.hour_to_interval_scaler = 60 / 5
        self.current_intervals = torch.zeros(
            self.num_paths, 
            dtype=self.dtype_int,
            device=self.device,
            requires_grad=False
        )
        self.max_interval = self.config.get("num_interval", 204) - 1

        # This variable will store the reward obtained at the most recent step
        # for each sample path.
        self.next_reward_by_class = torch.zeros(
            (self.S, self.num_paths), 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )
        self.total_reward_by_class = torch.zeros(
            (self.S, self.num_paths), 
            dtype=self.dtype_float,
            device=self.device,
            requires_grad=False
        )

        # ========== Get data from config ========== #
        self.lambda_5min = torch.tensor(self.config["lambda_5min"], dtype=self.dtype_float, device=self.device)
        self.lambda_hourly_by_interval = self.lambda_5min * self.hour_to_interval_scaler
        self.mu_hourly = torch.tensor(self.config["mu_hourly"], dtype=self.dtype_float, device=self.device)
        self.theta_hourly = torch.tensor(self.config["theta_hourly"], dtype=self.dtype_float, device=self.device)
        self.arr_cdf = torch.tensor(self.config["arr_cdf"], dtype=self.dtype_float, device=self.device)
        self.arr_pdf = torch.diff(self.arr_cdf, dim=1, prepend=torch.zeros(self.arr_cdf.shape[0], 1, dtype=self.dtype_float, device=self.device))
        self.cost_holding_hourly = torch.tensor(self.config["cost_holding_hourly"], dtype=self.dtype_float, device=self.device)
        self.cost_abandonment = torch.tensor(self.config["cost_abandonment"], dtype=self.dtype_float, device=self.device)
        self.cost_total_hourly = torch.tensor(self.config["cost_total_hourly"], dtype=self.dtype_float, device=self.device)
        self.num_server = torch.tensor(self.config["num_server"], dtype=self.dtype_int, device=self.device)
        self.num_server_init = torch.tensor(self.config["num_server_init"], dtype=self.dtype_int, device=self.device)

        # Initialize priority_order for each sample path
        self.priority_order = torch.zeros((self.num_paths, self.S), dtype=torch.int64, device=self.device)
        self._update_control_rules()

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

    def _update_transition_rates(self):
        """
        Compute the transition rates for each possible event, given the current state and the chosen action.
        Updates self.next_rates with shape (E, num_paths) containing:
        - rates[0] = arrival rates for each path
        - rates[1] = service completion rates for each path 
        - rates[2] = abandonment rates for each path
        Updates self.next_rates_total with shape (num_paths). 
        """
        # Reset rates
        self.next_rates.zero_()
        
        # Event type 0: Arrivals - sum arrival rates across all classes
        self.next_rates[0] = self.lambda_hourly_by_interval[self.current_intervals]
        # Event type 1: Service completions - sum mu * number in service for each class
        self.next_rates[1] = torch.sum(self.mu_hourly[:, None] * self.next_actions, dim=0)
        # Event type 2: Abandonments - sum theta * number waiting for each class
        self.next_rates[2] = torch.sum(self.theta_hourly[:, None] * self.waiting_customers, dim=0)
        # Update total rates
        self.next_rates_total = torch.sum(self.next_rates, dim=0)

        # Note that we need to ensure all next_rates_total are strictly positive
        if not torch.all(self.next_rates_total > 0):
            raise ValueError("All next_rates_total must be strictly positive.")

    def _update_rewards(self):
        """
        Compute the reward for each sample path, given the current state and action.
        Updates self.next_reward with shape (num_paths). 
        Updates self.total_reward with shape (num_paths). 
        """
        
        # Compute the reward for each sample path by class
        self.next_reward_by_class = -1.0 * self.cost_total_hourly[:, None] * self.waiting_customers
        self.next_reward_by_class *= self.next_time_increments[None, :]
        self.total_reward_by_class += self.next_reward_by_class

        self.next_reward = torch.sum(self.next_reward_by_class, dim=0)
        self.next_reward = self.next_reward
        self.total_reward += self.next_reward

    def _update_states(self):
        """
        Update the state based on which event occurred for each sample path.
        Updates self.current_states with shape (S, num_paths). 
        """
        # Event type 0: Arrivals
        arrival_mask = (self.next_event_indices == 0)
        # Sample which class arrives based on arrival probabilities
        arrival_probs = self.arr_pdf[self.current_intervals[arrival_mask]]
        arrival_classes = torch.multinomial(arrival_probs, num_samples=1).squeeze()
        # Increment state for the sampled class
        self.current_states[arrival_classes, arrival_mask] += 1

        # Event type 1: Service completions
        service_mask = (self.next_event_indices == 1)
        # Sample which class departs based on service rates
        departure_rates = (self.mu_hourly[:, None] * self.next_actions[:, service_mask]).t()
        departure_probs = departure_rates / (torch.sum(departure_rates, dim=1, keepdim=True) + 1e-10)
        departure_class = torch.multinomial(departure_probs, num_samples=1).squeeze()
        # Decrement state for the sampled class
        self.current_states[departure_class, service_mask] -= 1

        # Event type 2: Abandonments
        abandon_mask = (self.next_event_indices == 2)
        # Sample which class abandons based on waiting customers
        abandon_probs = (self.theta_hourly[:, None] * self.waiting_customers[:, abandon_mask]).t()
        abandon_probs = abandon_probs / (torch.sum(abandon_probs, dim=1, keepdim=True) + 1e-10)
        abandon_class = torch.multinomial(abandon_probs, num_samples=1).squeeze()
        # Decrement state for the sampled class
        self.current_states[abandon_class, abandon_mask] -= 1

    def _summarize(self) -> None:
        """
        Summarize the current batch of states, times, and rewards, 
        and record them in the history.
        """
        # # Save state, time, and last reward
        # self.history.append({
        #     "state": self.current_states.clone().cpu(),
        #     "time": self.current_times.clone().cpu(),
        #     "reward": self.total_reward.clone().cpu()
        # })
        pass


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
    simulator = CallCenterSimulator(
        config=config,
        num_paths=10000,
        device=device,
        seed=42
    )

    simulator.run_until_time(target_time=17)

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