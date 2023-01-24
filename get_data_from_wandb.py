def getRuns(api, projectName):
  print('Getting runs for project: ' + projectName)
  runs = api.runs(projectName)
  return runs

def getMetrics(runs, metric='epoch'):
  print('Getting metrics for runs')
  metrics = []
  names = []
  for run in runs:
    print(run.summary._json_dict.keys())
    values = []
    for row in run.scan_history(page_size=10000, min_step=1, max_step=10000, keys=[metric]):
      values.append(row[metric])

    metrics.append(values)
    names.append(run.name)
  return metrics, names

def saveMetrics(metrics, project_name, filenames):
  import os  
  os.makedirs(f'./models_performance/{project_name}', exist_ok=True)

  for i in range(len(metrics)):
    metrics[i].to_csv(f'./models_performance/{project_name}/{filenames[i]}.csv')

def __main__():
  import wandb
  api = wandb.Api()
  print('Initialized API')
  PROJECT_NAMES = ['MNIST_CNN', 'MNIST_FC', 'CIFAR_CNN', 'CIFAR_FC']
  for project_name in PROJECT_NAMES:
    runs = getRuns(api, project_name)
    metrics, model_names = getMetrics(runs)
    saveMetrics(metrics, project_name, model_names)

__main__()