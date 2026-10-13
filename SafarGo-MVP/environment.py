import numpy as np


class SafarGoEnv:
    """Synthetic aggregate simulator for 40 zones and five-minute decisions."""

    def __init__(self, zones=40, vehicles=300, horizon=288, seed=7):
        self.zones = zones
        self.vehicles = vehicles
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.base_demand = self.rng.uniform(0.15, 1.0, zones)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.supply = np.full(self.zones, self.vehicles // self.zones, dtype=int)
        self.supply[: self.vehicles % self.zones] += 1
        self.served_by_zone = np.zeros(self.zones, dtype=int)
        self.total = {
            "rides": 0, "revenue_pkr": 0.0, "wait_sum": 0.0,
            "requests": 0, "fuel_units": 0.0, "cancelled": 0,
        }
        return self.state()

    def demand(self):
        hour = (self.t % 288) // 12
        rush = 1.45 if 7 <= hour <= 9 or 16 <= hour <= 19 else 1.0
        evening = 1.25 if 20 <= hour <= 23 else 1.0
        weather = 1.25 if (self.t // 36) % 5 == 2 else 1.0
        pulse = np.ones(self.zones)
        pulse[(hour * 3 + np.arange(self.zones)) % 17 == 0] = 1.8
        expected = self.base_demand * rush * evening * weather * pulse * 4.5
        return self.rng.poisson(expected).astype(int)

    def state(self):
        hour = (self.t % 288) // 12
        weather_flag = int((self.t // 36) % 5 == 2)
        imbalance = int(np.clip(round((self.supply.mean() - self.supply.min()) / 2), 0, 4))
        avg_requests = self.total["requests"] / max(1, self.t)
        demand_bucket = int(np.clip(round(avg_requests / 10), 0, 4)) if self.t else 2
        return (hour // 4, weather_flag, imbalance, demand_bucket)

    def step(self, action):
        action = int(np.clip(action, 0, self.zones))
        fuel = 0.0
        if action < self.zones:
            donor = int(np.argmax(self.supply))
            if donor != action and self.supply[donor] > 1:
                self.supply[donor] -= 1
                self.supply[action] += 1
                fuel = 1.0 + abs(donor - action) / self.zones * 3.0

        requests = self.demand()
        completed = np.minimum(requests, self.supply)
        unmet = np.maximum(0, requests - self.supply)
        wait_by_zone = np.where(requests > 0, 2.0 + 7.0 * unmet / np.maximum(1, requests), 0.0)
        wait_sum = float((wait_by_zone * requests).sum())
        cancellations = int(unmet.sum())
        rides = int(completed.sum())
        request_count = int(requests.sum())
        revenue_pkr = rides * 300.0  # illustrative fare proxy, not a market estimate

        # Trips finish within a decision cycle in this simplified simulator. The
        # same vehicles return to their zone counts; rides never create vehicles.
        self.served_by_zone += completed
        fairness_gap = float(np.std(self.served_by_zone / max(1.0, self.served_by_zone.mean())))
        avg_wait = wait_sum / max(1, request_count)
        reward = (rides * 2.0 - max(0.0, avg_wait - 7.0) * 1.5
                  - fuel * 0.8 - cancellations * 0.5 - fairness_gap * 0.15)

        self.total["rides"] += rides
        self.total["revenue_pkr"] += revenue_pkr
        self.total["wait_sum"] += wait_sum
        self.total["requests"] += request_count
        self.total["fuel_units"] += fuel
        self.total["cancelled"] += cancellations
        self.t += 1
        done = self.t >= self.horizon
        info = {"rides": rides, "requests": request_count, "wait_min": avg_wait,
                "fuel_units": fuel, "revenue_pkr": revenue_pkr, "cancelled": cancellations}
        return self.state(), reward, done, info

    def metrics(self):
        return {
            "Completed rides": round(self.total["rides"], 1),
            "Revenue (PKR)": round(self.total["revenue_pkr"], 0),
            "Average wait (min)": round(self.total["wait_sum"] / max(1, self.total["requests"]), 2),
            "Fuel proxy (units)": round(self.total["fuel_units"], 1),
            "Cancelled requests": round(self.total["cancelled"], 1),
            "Zones served": int(np.count_nonzero(self.served_by_zone)),
        }
