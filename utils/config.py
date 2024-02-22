# let parameter can be used in every module

# train parameter reset
def initialize():
    global best_metric, best_metric_epoch, metric_values, epoch_loss_values
    best_metric = 0
    best_metric_epoch = 0
    metric_values = list()
    epoch_loss_values = list()