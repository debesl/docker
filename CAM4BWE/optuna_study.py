import optuna
from train import *

def objective(trial):

    lr = trial.suggest_float("lr",1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 16])
    history = main(lr, weight_decay, batch_size)
    return float(min(history.history["val_loss"]))


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="bandwidth_extension_study",
        direction="minimize",
        storage="sqlite:///optuna_study_mlp_stft.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial val_loss:", study.best_value)
    print("Best params:", study.best_params)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))