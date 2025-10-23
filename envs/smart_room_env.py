import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:
    import envs.utils as utils


class SmartRoomEnv(gym.Env):
    """
    Gym environment custom cho phòng học thông minh:
    State: [T, L, O, T_out, Dt]
    Dt={f1,f2,f3, ac1,ac2, light1,light2}
    Action: MultiDiscrete [3,3,3, 3,3, 2,2] = (f1,f2,f3, c1,c2, d1,d2)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 scenario_csv: str = None,
                 dt_minutes: float = 5.0,
                 heat_trans_rate: float = 0.05, people_heat_gain: float = 0.01,
                 cooling_effect=(0.0, 0.5, 1.0),
                 lamp_lux=(200.0, 200.0),
                 p_user_override: float = 0.02,
                 user_override_enabled: bool = True,
                 max_steps: int = 96,
                 c_energy: float = 0.01,
                 c_temp: float = 5.0,
                 c_light: float = 2.0,
                 c_switch: float = 0.1,
                 T_target: float = 24.0,
                 delta_T: float = 1.0,
                 L_target: float = 300.0,
                 noise_std: float = 0.0):
        super().__init__()

        self.dt_minutes = dt_minutes
        self.dt = dt_minutes / 60.0
        self.heat_trans_rate = heat_trans_rate
        self.people_heat_gain = people_heat_gain
        self.cooling_effect = np.array(cooling_effect, dtype=np.float32)
        self.lamp_lux = np.array(lamp_lux, dtype=np.float32)

        # override
        self.p_user_override = p_user_override
        self.user_override_enabled = user_override_enabled
        self.max_steps = max_steps

        # hệ số phần thưởng
        self.c_energy = c_energy
        self.c_temp = c_temp
        self.c_light = c_light
        self.c_switch = c_switch
        self.T_target = T_target
        self.delta_T = delta_T
        self.L_target = L_target

        # nhiễu (chỉ dùng cho NoisyNet)
        self.noise_std = noise_std

        # công suất thiết bị
        self.P_fan = [0.0, 75.0, 110.0]
        self.P_lamp_group = 40.0
        self.P_ac = [0.0, 1000.0, 2000.0]

        # Action/Observation space
        self.action_space = spaces.MultiDiscrete([3,3,3, 3,3, 2,2])
        obs_low  = np.array([0.0,0.0,0.0,-10.0] + [0]*7, dtype=np.float32)
        obs_high = np.array([40.0,2000.0,100.0,50.0] + [2,2,2,2,2,1,1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Kịch bản
        self.scenario = None
        if scenario_csv is not None:
            self.scenario = pd.read_csv(scenario_csv)
            for c in ["T_out","L_nat","N"]:
                if c not in self.scenario.columns:
                    raise ValueError(f"Scenario must contain {c}")

        self.state = None
        self._step_count = 0
        self.last_device_state = np.zeros(7, dtype=int)
        self.reset()

    def _sample_user_action(self, current_D):
        if not self.user_override_enabled:
            return None
        new = current_D.copy()
        for i in range(len(new)):
            if np.random.rand() < self.p_user_override:
                new[i] = np.random.randint(0,2) if i>=5 else np.random.randint(0,3)
        return new

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        T0, L0, N0, T_out0 = 27.0, 300.0, 0, 30.0
        D0 = np.zeros(7, dtype=int)
        if self.scenario is not None and len(self.scenario) > 0:
            row = self.scenario.iloc[0]
            T_out0, L0, N0 = float(row["T_out"]), float(row["L_nat"]), int(row["N"])
        self.state = np.concatenate(([T0, L0, N0, T_out0], D0)).astype(np.float32)
        self.last_device_state = D0.copy()
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        T, L, N, T_out = self.state[:4]
        D_curr = self.state[4:].astype(int)

        if self.scenario is not None and len(self.scenario) > 0:
            row = self.scenario.iloc[self._step_count % len(self.scenario)]
            T_out = float(row["T_out"])
            L_nat = float(row["L_nat"])
            N = int(row["N"])
        else:
            L_nat = 200.0

        D_agent = np.array(action, dtype=int)
        D_user = self._sample_user_action(D_curr)
        if D_user is None:
            D_user = D_curr
        D_next = utils.apply_user_override(D_agent, D_user, D_curr)

        T_next = utils.update_temperature(
            T, T_out, N, D_next[3:5],
            self.heat_trans_rate, self.people_heat_gain, self.dt,
            self.cooling_effect
        )
        L_next = utils.update_light(L, D_next[5:], self.lamp_lux, L_nat)
        P_total = utils.compute_power(D_next, self.P_fan, self.P_ac, self.P_lamp_group)
        energy_kwh = P_total * self.dt_minutes / 60 / 1000

        reward, Dtemp, Dlight, S = utils.compute_reward(
            T_next, L_next, N,
            D_curr, D_next,
            energy_kwh,
            self.T_target, self.delta_T, self.L_target,
            self.c_energy, self.c_temp, self.c_light, self.c_switch
        )

        self.state = np.array([T_next, L_next, float(N), T_out] + list(D_next.astype(float)), dtype=np.float32)
        self._step_count += 1
        done = self._step_count >= self.max_steps

        info = {
            "power_W": P_total,
            "energy_kwh": energy_kwh,
            "discomfort_temp": Dtemp,
            "discomfort_light": Dlight,
            "switch_count": S
        }
        return self._get_obs(), float(reward), done, False, info

    def _get_obs(self):
        obs = self.state.copy()
        if self.noise_std > 0:
            obs += np.random.normal(0, self.noise_std, size=obs.shape)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)

    def render(self):
        T, L, N, T_out = self.state[:4]
        D = self.state[4:].astype(int)
        print(f"step={self._step_count} | T={T:.2f}°C L={L:.1f}lux N={int(N)} Tout={T_out:.1f}°C | devices={D.tolist()}")
