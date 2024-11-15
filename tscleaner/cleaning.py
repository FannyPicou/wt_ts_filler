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
            if np.isnan(value) or np.isnan(prev_value):
                # "Value ok"
                data[t] = value
                prev_value = value
            if abs(value - prev_value) <= self.max_jump:
                # "Value ok"
                data[t] = value
                prev_value = value
            else:
                data[t] = np.nan
                print("Jump detected and value removed on", t, ":", value)
        print(f"Data removed: {data_original[~data_original.isin(data)]}")
        return data
        # return data.dropna()

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
            count = 0
            while data[i+count+1] == data[i+count] :
                count += 1
            if count >= self.flat_period :
                print("Removing ", count, "values starting at index:", i)
                data[i: i + count + 1] = np.nan
                i = i + 1 + count
            else:
                i += 1
        print(f"Data removed: {data_original[~data_original.isin(data)]}")
        return data