import json
import math
import random
import csv
from typing import List
import os
import datetime


class Simulation:
    """
    The Simulation class initializes configuration parameters from a JSON file
    and provides methods to read vectors/matrices from CSV files, as well as
    a method to run and save simulation results according to different priority rules.
    """

    def __init__(self, json_file_name: str, rule: str):
        """
        Constructor that parses the JSON configuration file and initializes
        all necessary parameters.

        :param json_file_name: Path to the JSON configuration file.
        :param rule: A string representing the priority rule (e.g., "cost", "c_mu_theta", etc.).
        :raises RuntimeError: If the JSON file cannot be opened or parsed.
        """
        # Load the JSON config
        try:
            with open(json_file_name, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise RuntimeError("Unable to open config.json")
        except Exception as e:
            raise RuntimeError(f"JSON parsing error: {str(e)}")

        # Accessing configuration values
        self.class_no = config["class_no"]
        self.num_interval = config["num_interval"]
        self.num_iterations = config["num_iterations"]
        self.priority_rule = rule

        # Paths
        lambda_path = config["lambda_path"]
        agents_path = config["agents_path"]
        mu_hourly_path = config["mu_hourly_path"]
        theta_hourly_path = config["theta_hourly_path"]
        arr_cdf_path = config["arr_cdf_path"]
        holding_cost_rate_path = config["holding_cost_rate_path"]
        abandonment_cost_rate_path = config["abandonment_cost_rate_path"]
        cost_rate_path = config["cost_rate_path"]
        initialization_path = config["initialization_path"]

        # Reading data from CSV files
        self.lambda_ = self.read_vector_from_csv(lambda_path)
        agents = self.read_vector_from_csv(agents_path)
        self.no_server = [int(val) for val in agents]

        self.mu_hourly = self.read_vector_from_csv(mu_hourly_path)
        self.theta_hourly = self.read_vector_from_csv(theta_hourly_path)
        self.arr_cdf = self.read_matrix_from_csv(arr_cdf_path)
        self.holding_cost_rate = self.read_vector_from_csv(holding_cost_rate_path)
        self.abandonment_cost_rate = self.read_vector_from_csv(abandonment_cost_rate_path)
        self.initialization = self.read_vector_from_csv(initialization_path)
        self.cost_rate = self.read_vector_from_csv(cost_rate_path)

        # For pathwise optimal benchmarks of high-dimensional test problems
        if self.class_no in [30, 50, 100]:
            # Extend mu_hourly and theta_hourly if needed
            if len(self.mu_hourly) < self.class_no:
                if len(self.mu_hourly) > 0:
                    fill_val = self.mu_hourly[0]
                else:
                    fill_val = 0
                self.mu_hourly += [fill_val] * (self.class_no - len(self.mu_hourly))

            if len(self.theta_hourly) < self.class_no:
                if len(self.theta_hourly) > 0:
                    fill_val = self.theta_hourly[0]
                else:
                    fill_val = 0
                self.theta_hourly += [fill_val] * (self.class_no - len(self.theta_hourly))

    def split_string(self, input_str: str, delimiter: str) -> List[str]:
        """
        Splits a string by the given delimiter.

        :param input_str: The input string to be split.
        :param delimiter: The character that delimits the input string.
        :return: A list of substring tokens.
        """
        return input_str.strip().split(delimiter)

    def read_matrix_from_csv(self, filename: str) -> List[List[float]]:
        """
        Reads a matrix (2D list) from a CSV file.

        :param filename: The path to the CSV file.
        :return: A list of lists (matrix) of float values.
        """
        matrix = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    row_str = self.split_string(line, ',')
                    row_floats = [float(x) for x in row_str]
                    matrix.append(row_floats)
        except FileNotFoundError:
            print(f"Failed to open the file: {filename}")
        return matrix

    def read_vector_from_csv(self, filename: str) -> List[float]:
        """
        Reads a vector (1D list) from a CSV file.

        :param filename: The path to the CSV file.
        :return: A list of float values.
        """
        vec = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    row_str = self.split_string(line, ',')
                    for cell in row_str:
                        vec.append(float(cell))
        except FileNotFoundError:
            print("Failed to open the file.")
        return vec

    def save(self, record_file: str) -> int:
        """
        Runs the simulation multiple times (based on 'num_iterations') and
        saves the results (cost per class and total cost) to 'record_file'.

        :param record_file: Path to the output CSV file.
        :return: 0 if successful.
        """
        # Open the file in write mode
        with open(record_file, 'w') as out_file:
            # Perform 'num_iterations' independent simulation runs
            for iteration in range(self.num_iterations):
                print(f"Iteration: {iteration}, Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
                # Create an Execute object
                exec_obj = Execute(
                    class_no_=self.class_no,
                    arr_cdf_=self.arr_cdf,
                    lambda_=self.lambda_,
                    mu_hourly_=self.mu_hourly,
                    theta_hourly_=self.theta_hourly,
                    no_server_=self.no_server,
                    holding_cost_rate_=self.holding_cost_rate,
                    abandonment_cost_rate_=self.abandonment_cost_rate,
                    cost_rate_=self.cost_rate,
                    num_interval_=self.num_interval,
                    seed=iteration,
                    priority_rule_=self.priority_rule
                )

                # Run the simulation and get cost results
                cost = exec_obj.run(self.initialization)

                # Prepare a CSV line: iteration, cost[0], cost[1], ..., cost[class_no], newline
                results_line = f"{iteration}," + ",".join(str(cost_val) for cost_val in cost) + "\n"
                out_file.write(results_line)

        return 0


class Execute:
    """
    The Execute class handles the main queueing simulation logic. It creates
    the events (arrival, departure, abandonment), updates system state, 
    and calculates performance metrics such as cost.
    """

    def __init__(
        self,
        class_no_: int,
        arr_cdf_: List[List[float]],
        lambda_: List[float],
        mu_hourly_: List[float],
        theta_hourly_: List[float],
        no_server_: List[int],
        holding_cost_rate_: List[float],
        abandonment_cost_rate_: List[float],
        cost_rate_: List[float],
        num_interval_: int,
        seed: int,
        priority_rule_: str
    ):
        """
        Constructor that initializes system parameters and random generator, and
        calls queue_init() to set up internal queue structures.

        :param class_no_: Number of classes.
        :param arr_cdf_: The arrival CDF for each time interval.
        :param lambda_: Arrival rates for each interval (per 5 minutes).
        :param mu_hourly_: Service rates (hourly) for each class.
        :param theta_hourly_: Abandonment rates (hourly) for each class.
        :param no_server_: Number of servers available in each interval.
        :param holding_cost_rate_: Hourly holding cost rate for each class.
        :param abandonment_cost_rate_: Hourly abandonment cost rate for each class.
        :param cost_rate_: Hourly cost rate for each class (used in queueing priority).
        :param num_interval_: The number of intervals in a day (or horizon).
        :param seed: The seed for the random number generator.
        :param priority_rule_: Priority rule name (e.g., "cost", "c_mu_theta", etc.).
        """
        self.class_no = class_no_
        self.arr_cdf = arr_cdf_
        self.lambda_ = lambda_
        self.no_server = no_server_
        self.mu_hourly = mu_hourly_
        self.theta_hourly = theta_hourly_
        self.holding_cost_rate = holding_cost_rate_
        self.abandonment_cost_rate = abandonment_cost_rate_
        self.cost_rate = cost_rate_
        self.num_interval = num_interval_
        self.priority_rule = priority_rule_

        # Random generator (Mersenne Twister equivalent in Python)
        self.generator = random.Random(seed)

        # Simulation states
        self.queue_list = []
        self.arr_list = []
        self.abandonment_list = []
        self.num_in_system = []
        self.num_in_service = []
        self.num_in_queue = []
        self.num_abandons = []
        self.queue_integral = []
        self.service_integral = []
        self.system_integral = []
        self.holding_cost = []
        self.waiting_cost = []
        self.total_cost = 0.0
        self.sim_clock = 0.0

        self.t_event = 0.0
        self.t_arrival = math.inf
        self.t_depart = math.inf
        self.t_abandon = math.inf

        self.class_abandon = 0
        self.cust_abandon = 0
        self.interval = 0
        self.pre_interval = 0
        self.post_interval = 0

        # Some constant or large number in place of numeric_limits<double>::max()
        self.inf = math.inf

        self.queue_init()

    def __del__(self):
        """Destructor-like method for cleanup (Python rarely needs this, but we include it for parity)."""
        pass

    def queue_init(self):
        """
        Initializes the queues for each class in the simulation.
        Each queue is represented as a list (in Python, we use a list or deque).
        """
        empty_queue = []
        not_an_empty_queue = [self.inf]  # This was used in C++ to store a sentinel for no empty queue

        # Reserve space
        self.queue_list = []
        self.arr_list = []
        self.abandonment_list = []

        for _ in range(self.class_no):
            self.queue_list.append([])          # analogous to pushing back empty_queue
            self.arr_list.append([])
            self.abandonment_list.append(not_an_empty_queue[:])  # copy of [inf]

    def generate_interarrival(self, interval: int) -> float:
        """
        Generates a random interarrival time based on an exponential distribution
        using the arrival rate at the given interval.

        :param interval: Index to select the per-5-min arrival rate from 'lambda_'.
        :return: Random interarrival time (in hours, consistent with the rate).
        """
        conversion_factor = 12.0  # number of 5-min segments in 1 hour
        arrival_rate = self.lambda_[interval] * conversion_factor  # hourly arrival rate
        # Python's expovariate uses the rate (lambda) parameter directly
        return self.generator.expovariate(arrival_rate)

    def generate_abandon(self, cls: int) -> float:
        """
        Generates a random abandonment time based on an exponential distribution.

        :param cls: The class index used to select the abandonment rate from 'theta_hourly'.
        :return: Random abandonment time (in hours).
        """
        abandonment_rate = self.theta_hourly[cls]
        return self.generator.expovariate(abandonment_rate)

    def generate_service(self) -> float:
        """
        Generates a random service time based on an exponential distribution.
        The rate is calculated as the sum of (num_in_service for each class) * (mu_hourly for each class).

        :return: Random service time.
        """
        service_rate = 0.0
        for i in range(self.class_no):
            service_rate += self.num_in_service[i + 1] * self.mu_hourly[i]
        return self.generator.expovariate(service_rate)

    def queueing_discipline(self, num_in_system: List[int], interval: int) -> List[float]:
        """
        Determines a priority order for classes based on cost rates or 
        other heuristic formulas (depending on self.priority_rule).

        :param num_in_system: The number of items in the system for each class.
        :param interval: The current time interval (not used here but kept for parity).
        :return: A list of class indices sorted by priority (ascending order in practice).
        """
        mu_theta_diff = [0.0] * self.class_no
        c_mu_theta_diff = [0.0] * self.class_no
        c_mu_theta = [0.0] * self.class_no
        c_mu = [0.0] * self.class_no
        kappa = []

        # Calculate different ranking metrics
        for i in range(self.class_no):
            mu_theta_diff[i] = self.mu_hourly[i] - self.theta_hourly[i]
            c_mu_theta[i] = self.cost_rate[i] * self.mu_hourly[i] / self.theta_hourly[i] if self.theta_hourly[i] != 0 else 0.0
            c_mu[i] = self.cost_rate[i] * self.mu_hourly[i]
            c_mu_theta_diff[i] = self.cost_rate[i] * (self.mu_hourly[i] - self.theta_hourly[i])

        # Assign a 'kappa' value used for sorting (argsort)
        for i in range(self.class_no):
            if self.priority_rule == "cost":
                kappa.append(-1.0 * self.cost_rate[i])
            elif self.priority_rule == "c_mu_theta":
                kappa.append(-1.0 * c_mu_theta[i])
            elif self.priority_rule == "c_mu":
                kappa.append(-1.0 * c_mu[i])
            elif self.priority_rule == "c_mu_theta_diff":
                kappa.append(-1.0 * c_mu_theta_diff[i])
            elif self.priority_rule == "mu_theta_diff":
                kappa.append(-1.0 * mu_theta_diff[i])
            else:
                # Default to cost if no recognized rule is found
                kappa.append(-1.0 * self.cost_rate[i])

        # Return indices sorted by kappa
        priority_order = self.argsort(kappa)
        return priority_order

    def optimal_policy_calculation(self, interval: int):
        """
        Calculates and applies the optimal policy for preemptive resume scheduling:
        which classes get service and how many servers are allocated to each class.
        """
        # Initialize policy with zeros
        optimal_policy = [0] * self.class_no
        priority_order = self.queueing_discipline(self.num_in_system, interval)

        num_served = 0

        # Determine the policy: how many remain in queue vs. get served
        for idx in priority_order:
            avail_server_num = max(self.no_server[interval] - num_served, 0)
            # People that remain in queue for class idx
            optimal_policy[idx] = max(self.num_in_system[idx + 1] - avail_server_num, 0)
            # People that get served for class idx
            num_served += min(self.num_in_system[idx + 1], avail_server_num)

        # Apply the difference in queue vs. service for each class
        diff = [0] * self.class_no
        for i in range(self.class_no):
            diff[i] = self.num_in_queue[i + 1] - optimal_policy[i]
            # Update number in service
            self.num_in_service[i + 1] += diff[i]
            # Adjust total queue count
            self.num_in_queue[0] -= diff[i]
            self.num_in_queue[i + 1] -= diff[i]

            # If diff[i] >= 0, some people move from queue to service
            if diff[i] >= 0:
                for _ in range(diff[i]):
                    self.remove_queue(i)
                    self.t_depart = self.sim_clock + self.generate_service()

            # If diff[i] < 0, some people in service are preempted back to the queue
            else:
                for _ in range(abs(diff[i])):
                    self.add_queue(self.t_event, i)

        # Update total number of people in service
        self.num_in_service[0] = sum(self.num_in_service[1:])

    def handle_arrival_event(self, interval: int, cls: int, pre_interval: int, post_interval: int):
        """
        Handles logic when a new arrival event occurs:
        - Increase number in system/queue
        - Possibly move someone into service immediately
        - Update times (t_arrival for the next arrival)
        - If not enough servers, call the policy recalculation.
        """
        self.num_in_system[0] += 1
        self.num_in_system[cls + 1] += 1

        # Schedule next arrival
        self.t_arrival = self.sim_clock + self.generate_interarrival(interval)

        # Add the person to the queue
        self.num_in_queue[0] += 1
        self.num_in_queue[cls + 1] += 1
        self.add_queue(self.t_event, cls)

        # Check if enough servers are free
        if ((self.num_in_system[0] <= self.no_server[interval] and pre_interval == post_interval) or
            (self.num_in_service[0] < self.no_server[interval] and pre_interval == post_interval)):

            if self.num_in_service[0] < self.no_server[interval]:
                # Move one directly to service
                self.num_in_service[0] += 1
                self.num_in_service[cls + 1] += 1
                self.num_in_queue[0] -= 1
                self.num_in_queue[cls + 1] -= 1

                # Remove from queue list
                self.remove_queue(cls)
                # Schedule service completion
                self.t_depart = self.sim_clock + self.generate_service()

            elif self.num_in_system[0] == 0:
                self.t_depart = self.inf
        else:
            self.optimal_policy_calculation(interval)

    def handle_depart_event(self, interval: int, cls: int, pre_interval: int, post_interval: int):
        """
        Handles logic when a departure event occurs:
        - Reduce counts in system/service
        - Possibly move someone from queue to service
        - If not enough servers or if the intervals changed, recalc the policy.
        """
        self.num_in_system[0] -= 1
        self.num_in_system[cls + 1] -= 1
        self.num_in_service[0] -= 1
        self.num_in_service[cls + 1] -= 1

        if (self.num_in_system[0] < self.no_server[interval] and pre_interval == post_interval):
            # If queue is non-empty, bring next person into service
            if self.num_in_queue[0] > 0 and self.num_in_service[0] < self.no_server[interval]:
                self.num_in_service[0] += 1
                self.num_in_service[cls + 1] += 1
                self.num_in_queue[0] -= 1
                self.num_in_queue[cls + 1] -= 1
                self.remove_queue(cls)

            if self.num_in_system[0] > 0:
                self.t_depart = self.sim_clock + self.generate_service()
            else:
                self.t_depart = self.inf
        else:
            self.optimal_policy_calculation(interval)

    def handle_abandon_event(self, interval: int, pre_interval: int, post_interval: int):
        """
        Handles logic when a customer abandons (leaves) the queue:
        - Remove them from the system/queue
        - Update abandonment lists
        - Possibly recalc policy if system is congested or interval changed.
        """
        MaxTime = math.inf

        # One person leaves the system
        self.num_in_system[0] -= 1
        self.num_in_system[self.class_abandon + 1] -= 1

        # Track number of abandons
        self.num_abandons[0] += 1
        self.num_abandons[self.class_abandon + 1] += 1

        # Remove person from queue
        self.num_in_queue[0] -= 1
        self.num_in_queue[self.class_abandon + 1] -= 1

        # Remove from queue_list and abandonment_list
        self.queue_list[self.class_abandon].pop(self.cust_abandon)
        self.abandonment_list[self.class_abandon].pop(self.cust_abandon + 1)

        # Recalculate next abandonment time
        if self.num_in_queue[0] > 0:
            min_abandon_times = [MaxTime] * self.class_no
            for i in range(self.class_no):
                if len(self.abandonment_list[i]) != 1:
                    # find min of abandonment_list[i][1:]
                    min_abandon_times[i] = min(self.abandonment_list[i][1:])

            nextAbandonTime = min(min_abandon_times)
            self.t_abandon = nextAbandonTime

            # Find which class and which customer
            for i in range(self.class_no):
                if nextAbandonTime in self.abandonment_list[i]:
                    self.class_abandon = i
                    # cust_abandon is (index in the list) - 1, since index 0 is inf
                    self.cust_abandon = self.abandonment_list[i].index(nextAbandonTime) - 1
                    break
        else:
            self.t_abandon = MaxTime

        if (self.num_in_system[0] > self.no_server[interval] or pre_interval != post_interval):
            self.optimal_policy_calculation(interval)

    def argsort(self, array: List[float]) -> List[int]:
        """
        Returns the indices that would sort 'array' in ascending order.

        :param array: A list of float values.
        :return: A list of indices that sorts 'array'.
        """
        return sorted(range(len(array)), key=lambda i: array[i])

    def add_queue(self, arr_time: float, cls: int):
        """
        Adds a newly arrived customer to the queue and schedules their abandonment time.
        Then updates the global next abandonment (t_abandon).

        :param arr_time: The arrival time of the new customer.
        :param cls: The class of the arriving customer.
        """
        MaxTime = math.inf

        # Add the new arrival to the queue
        self.queue_list[cls].append(arr_time)
        self.abandonment_list[cls].append(arr_time + self.generate_abandon(cls))

        # Find the global min across all classes
        min_temp = [MaxTime] * self.class_no
        for i in range(self.class_no):
            if len(self.abandonment_list[i]) != 1:
                min_temp[i] = min(self.abandonment_list[i])

        min_abandon = min(min_temp)
        self.t_abandon = min_abandon

        # Find which class and customer will abandon next
        for i in range(self.class_no):
            if min_abandon in self.abandonment_list[i]:
                self.class_abandon = i
                self.cust_abandon = self.abandonment_list[i].index(min_abandon) - 1
                break

    def remove_queue(self, cls: int):
        """
        Removes the first customer from the specified class queue and updates the
        next abandonment time across all classes.

        :param cls: The class index from which a customer is removed.
        """
        MaxTime = math.inf
        # Remove from front of the queue
        if len(self.queue_list[cls]) > 0:
            self.queue_list[cls].pop(0)
        if len(self.abandonment_list[cls]) > 1:
            self.abandonment_list[cls].pop(1)

        # Recalculate next abandonment
        if self.num_in_queue[0] > 0:
            min_temp = [MaxTime] * self.class_no
            for i in range(self.class_no):
                if len(self.abandonment_list[i]) > 1:
                    min_temp[i] = min(self.abandonment_list[i][1:])
            min_abandon = min(min_temp)
            self.t_abandon = min_abandon

            # Identify which class is next to abandon
            for i in range(self.class_no):
                if min_abandon in self.abandonment_list[i]:
                    self.class_abandon = i
                    break
            # Identify which customer
            cust_itr_index = self.abandonment_list[self.class_abandon][1:].index(min_abandon)
            self.cust_abandon = cust_itr_index
        else:
            self.t_abandon = MaxTime

    def run(self, initialization: List[float]) -> List[float]:
        """
        Main simulation loop that runs until sim_clock reaches T. Tracks queue lengths,
        service usage, and costs.

        :param initialization: A list indicating initial number of people in service for each class.
        :return: A list of cost values, where indices 0..class_no-1 are the cost per class
                 and index class_no is the total cost.
        """
        # We assume T is 17 hours, consistent with the logic in C++
        T = 17.0
        MaxTime = math.inf
        overtime_cost = 2.12

        # Initialize counters
        self.num_in_system = [0] * (self.class_no + 1)
        self.num_in_service = [0] * (self.class_no + 1)

        # Initialize the system state
        for i in range(self.class_no):
            self.num_in_system[i + 1] = int(initialization[i])
            self.num_in_service[i + 1] = int(initialization[i])

        # If we have initial agents, the total in system is the sum of all classes
        self.num_in_service[0] = sum(self.num_in_service[1:])
        self.num_in_system[0] = sum(self.num_in_system[1:])

        self.num_arrivals = [0] * (self.class_no + 1)
        self.num_in_queue = [0] * (self.class_no + 1)
        self.num_abandons = [0] * (self.class_no + 1)
        self.queue_integral = [0.0] * (self.class_no + 1)
        self.service_integral = [0.0] * (self.class_no + 1)
        self.system_integral = [0.0] * (self.class_no + 1)
        self.holding_cost = [0.0] * self.class_no
        self.waiting_cost = [0.0] * self.class_no
        self.total_cost = 0.0
        self.interval = 0
        self.sim_clock = 0.0

        # Initialize event times
        self.t_arrival = self.generate_interarrival(self.interval)
        self.t_depart = MaxTime
        self.t_abandon = MaxTime

        # Main simulation loop
        while self.sim_clock < T:
            self.t_event = min(self.t_arrival, self.t_depart, self.t_abandon)

            # Integrate queue, service, and system
            dt = self.t_event - self.sim_clock
            for i in range(self.class_no + 1):
                self.queue_integral[i] += self.num_in_queue[i] * dt
                self.service_integral[i] += self.num_in_service[i] * dt
                self.system_integral[i] += self.num_in_system[i] * dt

            # Integrate costs
            for i in range(self.class_no):
                self.holding_cost[i] += self.num_in_queue[i + 1] * self.holding_cost_rate[i] * dt
                self.waiting_cost[i] += self.num_in_queue[i + 1] * self.cost_rate[i] * dt
                self.total_cost += self.num_in_queue[i + 1] * self.cost_rate[i] * dt

            self.sim_clock = self.t_event
            self.pre_interval = self.interval
            # interval is int(sim_clock*12), but capped at 203
            self.interval = min(int(self.sim_clock * 12), 203)
            self.post_interval = self.interval

            # Determine which event triggered
            if math.isclose(self.t_event, self.t_arrival):
                # Figure out which class arrived, based on arr_cdf
                arrival_seed = self.generator.uniform(0.0, 1.0)
                cdf_list = self.arr_cdf[self.interval]
                # Find index where arrival_seed fits
                arrival_ind = 0
                for idx, val in enumerate(cdf_list):
                    if arrival_seed <= val:
                        arrival_ind = idx
                        break
                self.handle_arrival_event(self.interval, arrival_ind, self.pre_interval, self.post_interval)

            elif math.isclose(self.t_event, self.t_depart):
                # Determine departing class
                departure_seed = self.generator.uniform(0.0, 1.0)
                numerator = [0.0] * self.class_no
                for i in range(self.class_no):
                    numerator[i] = self.num_in_service[i + 1] * self.mu_hourly[i]
                service_rate_sum = sum(numerator)

                # Build cdf
                ser_cdf = []
                running_sum = 0.0
                for i in range(self.class_no):
                    fraction = numerator[i] / service_rate_sum if service_rate_sum > 0 else 0
                    running_sum += fraction
                    ser_cdf.append(running_sum)

                service_ind = 0
                for idx, val in enumerate(ser_cdf):
                    if departure_seed <= val:
                        service_ind = idx
                        break

                self.handle_depart_event(self.interval, service_ind, self.pre_interval, self.post_interval)

            elif math.isclose(self.t_event, self.t_abandon):
                self.handle_abandon_event(self.interval, self.pre_interval, self.post_interval)
            else:
                print("Something is Wrong")

        # Gather results: cost per class + total cost
        res = [0.0] * (self.class_no + 1)
        for i in range(self.class_no):
            res[i] = self.waiting_cost[i] + overtime_cost * self.num_in_queue[i + 1]
        res[self.class_no] = self.total_cost + overtime_cost * self.num_in_queue[0]
        return res


def main():
    """
    Main function to replicate the original C++ 'main'. 
    It defines the priority rules, reads the config file, 
    and runs the simulations for each rule.
    """
    # rules = ["cost", "c_mu_theta", "c_mu_theta_diff", "c_mu", "mu_theta_diff"]
    # json_file_name = "configs/call_center/config_17dim/config_17dim.json"

    # for rule_const in rules:
    #     rule = rule_const
    #     record_file = f"temp/static_benchmark_{rule}.csv"
    #     sim_obj = Simulation(json_file_name, rule)
    #     sim_obj.save(record_file)
    #     print(f"Simulation completed for rule: {rule}")

    json_file_name = "configs/call_center/config_17dim/config_17dim.json"
    sim_obj = Simulation(json_file_name, "c_mu_theta")
    sim_obj.save("temp/static_benchmark_c_mu_theta.csv")

if __name__ == "__main__":
    main()
