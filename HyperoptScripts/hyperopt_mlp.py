import math
import os
import sys
sys.path.append(os.path.dirname(sys.argv[0])+"/../TensorFlowScripts")

import train_GED

from hyperopt import hp, fmin, rand, tpe, STATUS_OK, Trials

space = {
    'lr': hp.loguniform('lr', math.log(1e-5), math.log(30)),
    'wd' : hp.choice('wd', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0.0]),
    'hsize': hp.choice('hsize', [2048]),
    'nlayers': hp.choice('nlayers', [3]),
    'Lambda': hp.choice('Lambda', [0.0]),
    'gamma': hp.choice('gamma', [0.999])
}

def objective(params):
    train_loss,val_loss = train_GED.main(params)
    print "LOSSES: ",train_loss,val_loss
    return {
        'loss': train_loss,
        'loss_variance': train_loss * (1.0 - train_loss),
        'true_loss': val_loss,
        'true_loss_variance': val_loss * (1.0 - val_loss),
        'status': STATUS_OK,
        # # -- store other results like this
        # 'eval_time': time.time(),
        # 'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # # -- attachments are handled differently
        # 'attachments':
        # {'time_module': pickle.dumps(time.time)}
    }

trials = Trials()
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)
print best
