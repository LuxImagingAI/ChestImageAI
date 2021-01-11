import numpy as np
import torch
import sklearn.metrics


def batch_prediction(model, loader, max_batch=-1, tta_ensemble = 1, device=None):
    
    model.eval()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble, targets = [], []
    for i in range(tta_ensemble):
        preds, targs = [], []
        with torch.no_grad():
            for i, (x, t, m) in enumerate(loader):
                x, t = x.to(device), t.numpy()
                if getattr(model, 'meta_injection', None):
                    m = m.to(device)
                    logits = model(x, m)
                else:
                    logits = model(x)
                preds.append(logits.to('cpu').numpy())
                targs.append(t)
                if i == max_batch:
                    break

        ensemble.append(np.vstack(preds))
        targets.append(np.vstack(targs))
    
    assert np.all(targets[0] == np.array(targets).mean(axis=0)), 'Targets across the ensemble do not match'
    
    return np.array(ensemble).squeeze(), targets[0]


def ensemble_mean(ensemble):
    return ensemble if len(ensemble.shape)<3 else ensemble.mean(axis=0)


def eval_auc(preds, targets, columns=None):
    include = targets >=0
    if columns is not None:
        include *= np.array([columns])
    return sklearn.metrics.roc_auc_score(targets[include], preds[include])

















def eval_mcc(model, loader, num_iter=-1):
    
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preds, targets = [], []
    with torch.no_grad():
        for i, (x, t) in enumerate(loader):
            x, t = x.to(device), t.numpy()
            logits = model(x)
            preds.append(logits.to('cpu').numpy())
            targets.append(t)
            if i == num_iter:
                break
    
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    ignore_index = targets >= 0
    
    return sklearn.metrics.matthews_corrcoef(targets[ignore_index], preds[ignore_index]>0)


def eval_crit(model, loader, crit, num_iter=-1, device=None):
    
    model.eval()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss, n_samples = 0, 0
    with torch.no_grad():
        for i, (x, y, m) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if getattr(model, 'meta_injection', None):
                m = m.to(device)
                logits = model(x, m)
            else:
                logits = model(x)
            l, n = crit(logits, y)
            loss += l.cpu().numpy()
            n_samples += n.cpu().numpy()
            if i == num_iter:
                break

    return loss / n_samples