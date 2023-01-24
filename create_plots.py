from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read the data
def read_data(model_name, filename):
    df = pd.read_csv(f'./models_performance/{model_name}/{filename}.csv')
    return df

def clean_proj_data(df):
  df['d_units'] = df['d_units'].apply(lambda row: sum(literal_eval(row)))

  return df[df.projected == True]

# plot accuracy comparison
def plot_line(df, title, y_threshold, xvalues="d_units", yvalues="val_acc", x_label="d", y_label="Accuracy"):
  #df = df.sort_values(by=[xvalues,], ascending=True)
  df = df.sort_values(by=[xvalues, yvalues], ascending=True)
  df = df.drop_duplicates(subset=xvalues, keep='last')
  print(df[xvalues])
  print(df[yvalues])
  axes = df.plot.line(x=xvalues, y=yvalues, title=title)
  axes.set_xlabel(x_label)
  axes.set_ylabel(y_label)
  axes.axhline(y = y_threshold, color = 'b', linestyle = ':')
  # Add the text '90%' to the plot
  axes.text(100, 1.003 * y_threshold, '90%')

  axes.legend(loc='lower right')

def plot_bar(df, title, xvalues="d_units", yvalues="val_acc", x_label="d", y_label="Accuracy"):
  df = df.sort_values(by=[yvalues], ascending=True)
  #df = df.drop_duplicates(subset=xvalues, keep='last')
  #df = df.loc[df["Tags"].str.contains('test_d_tuning')]
  #df[xvalues] = df[xvalues].astype(str)
  #df.loc[df['Tags'].str.contains('sparse_coo'), xvalues] = 'sparse_coo'

  if(yvalues == "Train_Allocated_VRAM"):
    df[yvalues] = df[yvalues] / (1024 * 1024)
  print(df[xvalues])
  print(df[yvalues])
  axes = df.plot.bar(x=xvalues, y=yvalues, title=title, rot=45)
  axes.set_xlabel(x_label)
  axes.set_ylabel(y_label)
  
  #axes.set_yticks(np.arange(0, 161, 10))
  #axes.set_yticks(np.arange(0, 161, 5), minor=True)

  axes.set_ylim(0.3, 0.5)
  axes.set_yticks(np.arange(0.3, 0.5, 0.005), minor=True)

  # set figure size
  axes.figure.set_size_inches(6,9)

  # move legend to the top left
  #axes.legend([y_label,], loc='upper left')
  # disble legend
  axes.legend().set_visible(False)

def plot_mult_lines(df_list, title, xvalues="d_units", yvalues="val_acc",     x_label="d", y_label="Accuracy", y_threshold=None, legend_names=None):
  axes = None
  for df in df_list:
    df = df.sort_values(by=[xvalues, yvalues], ascending=True)
    df = df.loc[df.d_units < 8000]
    df = df.drop_duplicates(subset=xvalues, keep='last')
    axes = df.plot.line(x=xvalues, y=yvalues, title=title, ax=axes)
  if(y_threshold):
    axes.axhline(y = y_threshold, color = 'b', linestyle = ':')
    # Add the text '90%' to the plot
    axes.text(1000, 1.003 * y_threshold, '90%')
  
  axes.set_xlabel(x_label)
  axes.set_ylabel(y_label)
  axes.legend(legend_names, loc='lower right')

'''
fn = "cifar_fc_sparse_vs_dense_runt"

df = read_data('CIFAR-FC-SparseVsDense', 'sparse_vs_dense_runt')

plot_bar(df, "CIFAR FC Sparse Vs Dense", xvalues="rp_gen_algorithm", yvalues="Runtime", x_label="Random Projection Type", y_label="Total Runtime")
'''


fn = "cifar_fc_sparse_vs_dense"

df = read_data('CIFAR-FC-SparseVsDense', 'sparse_vs_dense_mem')

plot_bar(df, "CIFAR FC Sparse Vs Dense", xvalues="rp_gen_algorithm", yvalues="val_acc", x_label="Random Projection Type", y_label="Accuracy")


'''
fn = "cifar_fc_sparse_vs_dense_mem"

df = read_data('CIFAR-FC-SparseVsDense', 'sparse_vs_dense_mem')

plot_bar(df, "CIFAR FC Sparse Vs Dense", xvalues="rp_gen_algorithm", yvalues="Train_Allocated_VRAM", x_label="Random Projection Type", y_label="VRAM(MB)")
'''

'''
fn = "cifar_fc_sparse_vs_dense_time"

df = read_data('CIFAR-FC-SparseVsDense', 'sparse_vs_dense_mem')

plot_bar(df, "CIFAR FC Sparse Vs Dense", xvalues="rp_gen_algorithm", yvalues="train_time_step", x_label="Random Projection Type", y_label="Time per Train Step (ms)")
'''

""" 
fn = "cifar_cnn_s_vs_cnn_62k_proj"

df = read_data('CIFAR-CNN-62k', 'summ_n_params_tuning')
df_cnn_s = read_data('CIFAR-CNN-S', 'summ_cifar_cnn_s')
df_cnn_s['d_units'] = df_cnn_s['Name'].apply(lambda row: int(row.split('_')[1]))

df_clean = clean_proj_data(df)
best_acc = df.loc[df.projected == False, 'val_acc'].max()

plot_mult_lines([df_clean, df_cnn_s], "CIFAR", y_threshold=best_acc*0.9, legend_names=['CNN-62k-Proj', 'Small-CNN']) 
"""
'''

fn = "mnist_cnn_s_vs_cnn_62k_proj"

df = read_data('MNIST-CNN-62k', 'summ_n_params_tuning')
df_cnn_s = read_data('MNIST-CNN-S', 'summ_mnist_cnn_s')
df_cnn_s['d_units'] = df_cnn_s['Name'].apply(lambda row: int(row.split('_')[1]))
df_clean = clean_proj_data(df)
best_acc = df.loc[df.projected == False, 'val_acc'].max()

plot_mult_lines([df_clean, df_cnn_s], "MNIST", y_threshold=best_acc*0.9, legend_names=['CNN-62k-Proj', 'Small-CNN'])
'''

plt.savefig(f'./plots/{fn}.pgf')
plt.savefig(f'./plots/{fn}.png')