from tensorboardX import SummaryWriter
from datetime import datetime

class Logger:
    def __init__(self, savedir):
        self.writer = SummaryWriter(savedir+datetime.now().strftime('%B%d_%H:%M:%S'))
    
    def train_step(self, train_loss, step):
        self.writer.add_scalar('train_loss', train_loss, step)

    def eval_step(self, eval_loss, step):
        self.writer.add_scalar('eval_loss', eval_loss, step)
