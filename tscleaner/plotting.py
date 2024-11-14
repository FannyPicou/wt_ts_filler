import matplotlib.pyplot as plt


def plot_timeseries(data_original, data):
    plt.figure(figsize=(10, 5))
    plt.plot(data_original, ".", color="red")
    plt.plot(data, ".", color="green")
    plt.title(f"{data.name}")
    plt.show()
