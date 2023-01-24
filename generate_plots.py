import matplotlib.pyplot as plt
import pandas as pd

# read the data
def read_data(project_name, model_name):
    df = pd.read_csv(f'./models_performance/{project_name}/{model_name}.csv')
    return df

# plot accuracy comparison
def plot_accuracy_comparison_num_params(project_name, model_names):
    fig, ax = plt.subplots()
    for model_name in model_names:
        df = read_data(project_name, model_name)
        ax.plot(df['step'], df['val_accuracy'], label=model_name)
    ax.set(xlabel='Epoch', ylabel='Accuracy', title=f'Accuracy comparison for {project_name}')
    ax.legend()
    fig.savefig(f'./plots/{project_name}_accuracy_comparison.png')

# plot loss comparison
def plot_loss_comparison_num_params(project_name, model_names):
    fig, ax = plt.subplots()
    for model_name in model_names:
        df = read_data(project_name, model_name)
        ax.plot(df['step'], df['val_loss'], label=model_name)
    ax.set(xlabel='Epoch', ylabel='Loss', title=f'Loss comparison for {project_name}')
    ax.legend()
    fig.savefig(f'./plots/{project_name}_loss_comparison.png')

def __init__():
    import os
    os.makedirs('./plots', exist_ok=True)
    PROJECT_NAMES = ['MNIST_FC', ]
    MODEL_NAMES = ['FC_Full_159k_dp2',]
    for project_name in PROJECT_NAMES:
      for model_name in MODEL_NAMES:
        df = read_data(project_name, model_name)
        print(df.columns)
        df.val_acc.plot()
        plt.show()

__init__()