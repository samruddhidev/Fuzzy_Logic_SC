# 🌀 Fuzzy Smart Fan Controller

An animated, step-by-step visualization of a **Mamdani Fuzzy Inference System (FIS)** that controls a fan's speed based on room temperature. Built with Python, NumPy and matplotlib.

Made for **Soft Computing IA1** by Samruddhi Jain and Insiya Lokhandwala.

## What it shows

For each test temperature, the animation walks through all 7 stages of fuzzy inference:

1. **Input**: a crisp temperature reading arrives
2. **Fuzzify**: convert it to membership degrees (Cold / Comfortable / Hot)
3. **Rules**: compute the firing strength (α) of each rule
4. **Inference**: clip each output set at its rule's α (min implication)
5. **Aggregate**: merge the clipped sets into one fuzzy output (max aggregation)
6. **Defuzzify**: get a crisp fan speed using the centroid method
7. **Output**: drive the fan and add the point to the response curve

The dashboard also has a live thermometer, a spinning fan whose speed and colour follow the output, and the full temperature → fan speed response curve.

## Fuzzy system design

| Variable | Fuzzy sets | Shape |
|---|---|---|
| Temperature (10–45 °C) | Cold, Comfortable, Hot | Trapezoid, Triangle, Trapezoid |
| Fan speed (0–100 %) | Slow, Medium, Fast | Trapezoid, Triangle, Trapezoid |

**Rule base**

| # | IF temperature is | THEN fan speed is |
|---|---|---|
| R1 | Cold | Slow |
| R2 | Comfortable | Medium |
| R3 | Hot | Fast |

**Method:** Mamdani, with min implication, max aggregation and centroid defuzzification.

## Getting started

### Requirements
- Python 3.8+
- `numpy`
- `matplotlib`
- `ffmpeg` (optional, only for saving an MP4)

```bash
pip install numpy matplotlib
```

### Run the live animation
```bash
python fuzzy_smart_fan.py
```

### Save as a video
```bash
python fuzzy_smart_fan.py --save
```
This saves `fuzzy_smart_fan.mp4`. If ffmpeg isn't installed, it falls back to `fuzzy_smart_fan.gif`.

## Customize

Edit the settings at the top of the script:

```python
EXAMPLES = [21, 30, 38]   # test temperatures in °C
SPEED_UP = 1              # 1 = normal, 2 = twice as fast, ...
INTERVAL_MS = 35          # delay between frames (ms)
```

You can also change the membership functions in `IN_SETS` / `OUT_SETS` and the rules in `RULES`.

## Tech stack

Python · NumPy · matplotlib (`FuncAnimation`)

## Authors

- Samruddhi Jain ([@samruddhidev](https://github.com/samruddhidev))
