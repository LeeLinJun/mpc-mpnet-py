from dataset.dataset import get_loader
from networks.mpnet import MPNet

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import click
from tqdm import tqdm
from pathlib import Path

@click.command()
@click.option('--lr', default=1e-3, help='learning_rate')
@click.option('--epochs', default=5000, help='epochs')
@click.option('--batch', default=128, help='batch')
@click.option('--system', default='sst_envs')
@click.option('--model', default='acrobot_obs')
@click.option('--setup', default='default')
def train(lr, epochs, batch, system, model, setup):
    mpnet = MPNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4)
    env_vox = torch.from_numpy(np.load('{}/{}_env_vox.npy'.format(system, model))).float()
    if torch.cuda.is_available():
        mpnet = mpnet.cuda()
        env_vox = env_vox.cuda()
#     optimizer = torch.optim.Adagrad(mpnet.parameters(), lr=lr)
#     optimizer = torch.optim.SGD(mpnet.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(mpnet.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    train_loader, test_loader = get_loader(system, model, batch_size=batch, setup=setup)
    
    for i in tqdm(range(epochs+1)):
        print("epoch {}".format(i))
        train_loss = []
        mpnet.train()
        for data, label in tqdm(train_loader):
            ## prepare data
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            inputs = data[:,1:]
            envs = env_vox[(data[:, 0]).long()]
            ## execute
            optimizer.zero_grad()
            output = mpnet(inputs, envs)
#             loss = F.mse_loss(output, label)
            loss = F.l1_loss(output, label)
            loss.backward()
            # print(loss.item())
#             train_loss.append(loss.item())
            optimizer.step()
            # break
        scheduler.step(i)
#         print("train loss:{}".format(np.mean(train_loss)))
        for data, label in test_loader:
#             mpnet.eval()
            eval_loss = []
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()                
                inputs = data[:,1:]
                envs = env_vox[(data[:,0]).long()]
                output = mpnet(inputs, envs)
#                 loss = F.mse_loss(output, label)
                loss = F.l1_loss(output, label)
                eval_loss.append(loss.item())
        print("eval loss:{}".format(np.mean(eval_loss)))
        if i%1 == 0:
            Path("output/{}/{}".format(model, setup)).mkdir(parents=True, exist_ok=True)
            torch.save(mpnet.state_dict(), "output/{}/{}/ep{}".format(model, setup, i))


if __name__ == "__main__":
    train()
