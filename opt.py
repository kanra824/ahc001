import optuna
import subprocess

def objective(trial):
    start_tmp = trial.suggest_uniform('start_tmp', 0, 0.1)
    end_tmp = trial.suggest_uniform('end_tmp', start_tmp,  0.1)
    cmd = f'./all {start_tmp} {end_tmp}'
    print (cmd)
    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate()
    print(stdout, stderr)
    return int(stdout)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print(study.best_trial)
print()
print (study.best_params)
print (study.best_value)
