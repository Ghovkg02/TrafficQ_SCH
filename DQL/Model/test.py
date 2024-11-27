import traci
import numpy as np
import random
import timeit
import os

# Define traffic light phases based on environment.net.xml
NS_GREEN = 0
NS_YELLOW = 1
NSL_GREEN = 2
NSL_YELLOW = 3
EW_GREEN = 4
EW_YELLOW = 5
EWL_GREEN = 6
EWL_YELLOW = 7

class TrafficSimulation:
    def __init__(self, model, traffic_generator, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self.model = model
        self.traffic_gen = traffic_generator
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self.step = 0
        self.reward_per_episode = []
        self.queue_lengths_per_episode = []
        self.waiting_times = {}
        self.old_action = -1  # Track previous action to check for yellow phase trigger

    def run_simulation(self, episode_num):
        start_time = timeit.default_timer()
        self.traffic_gen.generate_routefile(seed=episode_num)
        traci.start(self.sumo_cmd)
        print(f"Starting simulation {episode_num}...")

        old_total_wait_time = 0
        self.step = 0

        while self.step < self.max_steps:
            current_state = self._gather_intersection_state()
            current_total_wait_time = self._calculate_total_wait_time()
            reward = old_total_wait_time - current_total_wait_time
            action = self._select_action(current_state)

            if self.step != 0 and self.old_action != action:
                self._switch_to_yellow(self.old_action)
                self._execute_phase(self.yellow_duration)

            self._switch_to_green(action)
            self._execute_phase(self.green_duration)

            self.old_action = action
            old_total_wait_time = current_total_wait_time
            self.reward_per_episode.append(reward)

        traci.close()
        simulation_duration = round(timeit.default_timer() - start_time, 1)
        return simulation_duration

    def _execute_phase(self, steps_remaining):
        if (self.step + steps_remaining) > self.max_steps:
            steps_remaining = self.max_steps - self.step

        for _ in range(steps_remaining):
            traci.simulationStep()
            self.step += 1
            queue_length = self._compute_queue_length()
            self.queue_lengths_per_episode.append(queue_length)

    def _calculate_total_wait_time(self):
        incoming_lanes = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for vehicle_id in traci.vehicle.getIDList():
            wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)

            if road_id in incoming_lanes:
                self.waiting_times[vehicle_id] = wait_time
            elif vehicle_id in self.waiting_times:
                del self.waiting_times[vehicle_id]

        return sum(self.waiting_times.values())

    def _select_action(self, state):
        return np.argmax(self.model.predict_one(state))

    def _switch_to_yellow(self, prev_action):
        yellow_phase = prev_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase)

    def _switch_to_green(self, action):
        green_phases = [NS_GREEN, NSL_GREEN, EW_GREEN, EWL_GREEN]
        traci.trafficlight.setPhase("TL", green_phases[action])

    def _compute_queue_length(self):
        halts = [traci.edge.getLastStepHaltingNumber(edge) for edge in ["N2TL", "S2TL", "E2TL", "W2TL"]]
        return sum(halts)

    def _gather_intersection_state(self):
        state = np.zeros(self.num_states)
        vehicles = traci.vehicle.getIDList()

        for vehicle in vehicles:
            lane_pos = 750 - traci.vehicle.getLanePosition(vehicle)
            lane_id = traci.vehicle.getLaneID(vehicle)
            lane_cell = self._determine_lane_cell(lane_pos)
            lane_group = self._determine_lane_group(lane_id)

            if lane_group is not None:
                position = int(f"{lane_group}{lane_cell}")
                state[position] = 1

        return state

    def _determine_lane_cell(self, lane_pos):
        if lane_pos < 7:
            return 0
        elif lane_pos < 14:
            return 1
        elif lane_pos < 21:
            return 2
        elif lane_pos < 28:
            return 3
        elif lane_pos < 40:
            return 4
        elif lane_pos < 60:
            return 5
        elif lane_pos < 100:
            return 6
        elif lane_pos < 160:
            return 7
        elif lane_pos < 400:
            return 8
        else:
            return 9

    def _determine_lane_group(self, lane_id):
        groups = {
            "W2TL_": 0, "N2TL_": 2, "E2TL_": 4, "S2TL_": 6
        }

        for group_prefix, group_num in groups.items():
            if lane_id.startswith(group_prefix):
                return group_num + int(lane_id[-1])

        return None

    @property
    def episode_rewards(self):
        return self.reward_per_episode

    @property
    def episode_queue_lengths(self):
        return self.queue_lengths_per_episode
