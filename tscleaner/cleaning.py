import numpy as np


class SpikeCleaner:
    def __init__(self, max_jump):
        self.max_jump = max_jump

    def clean(self, data):
        data_original = data.copy()
        print(f"Checking for jumps in {data.name}")
        prev_value = data.iloc[0]
        for t, value in data.items():
            if isinstance(value, (str)):  # raise exception for invalid input
                raise ValueError(f"The provided value of {value} is string type.")
            if abs(value - prev_value) <= self.max_jump:
                # "Value ok"
                data[t] = value
                prev_value = value
            else:
                data[t] = np.nan
                print("Jump detected and value removed on", t, ":", value)
        print(f"Data removed: {data_original[~data_original.isin(data)]}")
        return data.dropna()


class OutOfRangeCleaner:
    def __init__(self, min_val, max_val):
        self.min_value = min_val
        self.max_value = max_val

    def clean(self, data):
        data_original = data.copy()
        for t, value in data.items():
            if isinstance(value, (str)):  # raise exception for invalid input
                raise ValueError(f"The provided value of {value} is string type.")
            # print("Checking value on", t, ":", value)
            if self.min_value <= value <= self.max_value:
                pass
                # print("Value ok:", value)
            else:
                data[t] = np.nan
                print("Value removed:", value)
        print(f"Data removed: {data_original[~data_original.isin(data)]}")
        # print("Data1 after range check:", data1)
        return data.dropna()


class FlatPeriodCleaner:
    def __init__(self, flat_period):
        self.flat_period = flat_period

    def clean(self, data):
        data_original = data.copy()
        for t, value in data.items():
            if isinstance(value, (str)):  # raise exception for invalid input
                raise ValueError(f"The provided value of {value} is string type.")
        print(f"Checking for flat periods in {data.name}")
        i = 0
        while i < len(data) - self.flat_period:
            if len(set(data[i : i + self.flat_period + 1])) == 1:
                print("Removing flat period starting at index:", i)
                data[i : i + self.flat_period + 1] = np.nan
                i += self.flat_period
            else:
                i += 1
        print(f"Data removed: {data_original[~data_original.isin(data)]}")
        # print("Data1 after flat period check:", data1)
        return data.dropna()
