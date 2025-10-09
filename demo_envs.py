from envs import SmartRoomEnv
import csv

env = SmartRoomEnv()
obs, info = env.reset()
rows = []

for step in range(96):
    a = env.action_space.sample()
    s, r, terminated, truncated, info = env.step(a)
    rows.append([step, a.tolist(), float(s[0]), float(s[1]), int(s[2]),
                 info['power_W'], info['energy_kwh'], r, info['switch_count']])
    if terminated or truncated:
        break

with open("demo_log.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["step","action","T","L","N","power_W","energy_kwh","reward","switch"])
    w.writerows(rows)

print("Saved demo_log.csv")
