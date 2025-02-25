import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to extract area values from area reports
def matchArea(data, keys):
    return [float(re.search(f"{key}:\\s+(.+)", data).group(1)) for key in keys]

# Function to extract power values from power reports
def matchPower(data):
    vp = "([0-9\\.+-e]+)\\s[mnu]W"
    r = re.search(f"Total\\s+{vp}\\s+{vp}\\s+{vp}\\s+{vp}", data)
    return [float(val) for val in r.groups()]

# Collect area data
weights = []
areas = []
area_keys = ['Total cell area']

for file in sorted(glob.glob('logs/area_*.txt')):
    weight = int(re.search(r'area_(-?\d+)\.txt', file).group(1))
    with open(file) as f:
        try:
            value = matchArea(f.read(), area_keys)[0]
            weights.append(weight)
            areas.append(value)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Collect power data
power_weights = []
powers = []

for file in sorted(glob.glob('logs/power_*.txt')):
    weight = int(re.search(r'power_(-?\d+)\.txt', file).group(1))
    with open(file) as f:
        try:
            data = f.read()
            value = matchPower(data)[0]  # Total power
            power_weights.append(weight)
            powers.append(value)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Plot area results
plt.figure(figsize=(10, 6))
plt.scatter(weights, areas, alpha=0.7)
plt.title('Total Cell Area vs Weight Value')
plt.xlabel('Weight')
plt.ylabel('Total Cell Area (μm²)')
plt.grid(True)
plt.savefig('area_vs_weight.png')
plt.show()

# Plot power results
plt.figure(figsize=(10, 6))
plt.scatter(power_weights, powers, alpha=0.7, color='red')
plt.title('Total Power vs Weight Value')
plt.xlabel('Weight')
plt.ylabel('Total Power (mW)')
plt.grid(True)
plt.savefig('power_vs_weight.png')
plt.show()