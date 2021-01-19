import matplotlib.pyplot as plt

def plot_per_epoch(train, val, mode, save_as):
    plt.figure(figsize=(12,4))
    plt.plot(train)
    plt.plot(val)
    plt.legend([f'training {mode}', f'validation {mode}'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.savefig(save_as)