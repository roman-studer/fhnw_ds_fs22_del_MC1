import wandb

api = wandb.Api()
run = api.run('ryzash01/del_MC1/')

files = run.files()
for file in files:
    if file.name == 'output.log':
        file.delete()
#%%
