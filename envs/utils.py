import numpy as np

def compute_power(devices, P_fan, P_ac, P_lamp_group):
    """Tính công suất tiêu thụ"""
    fans = devices[0:3]
    acs = devices[3:5]
    lamps = devices[5:7]
    E_fans = sum(P_fan[int(l)] for l in fans)
    E_ac   = sum(P_ac[int(l)] for l in acs)
    E_lamps = float(np.sum(lamps)) * P_lamp_group
    return E_fans + E_ac + E_lamps

def switching_cost(prev_devices, new_devices):
    return int(np.sum(np.array(prev_devices) != np.array(new_devices)))

def discomfort_temp(T, N, T_target, delta_T):
    if N <= 0: return 0.0
    return max(0.0, abs(T - T_target) - delta_T)

def discomfort_light(L, N, L_target):
    if N <= 0: return 0.0
    return max(0.0, L_target - L)

def update_temperature(T, T_out, N, ac_levels, heat_trans_rate, people_heat_gain, dt, cooling_effect):
    cooling = np.sum(cooling_effect[ac_levels])
    return T + heat_trans_rate*(T_out-T)*dt + people_heat_gain*N - cooling + np.random.normal(0,0.05)

def update_light(L_prev, lamps, lamp_lux, scenario, step_idx):
    L_nat = float(scenario.iloc[min(step_idx, len(scenario)-1)]["L_nat"]) if scenario is not None else 200.0
    L_next = L_nat + np.sum(lamp_lux * lamps.astype(float)) + np.random.normal(0,1.0)
    return max(0.0, L_next)

def compute_reward(T_next, L_next, N, D_curr, D_next, P_total,
                   T_target, delta_T, L_target,
                   c_energy, c_temp, c_light, c_switch):
    S = switching_cost(D_curr, D_next)
    Dtemp = discomfort_temp(T_next, N, T_target, delta_T)
    Dlight = discomfort_light(L_next, N, L_target)
    reward = -(c_energy*P_total + c_temp*Dtemp + c_light*Dlight + c_switch*S)
    return reward, Dtemp, Dlight, S

def apply_user_override(D_agent, D_user, D_curr):
    D_next = D_agent.copy()
    for i in range(len(D_next)):
        if D_user[i] != D_curr[i]:
            D_next[i] = D_user[i]
    return D_next
