from dataset.dataset import get_loader_cost
from networks.mpnet import MPNet
from networks.costnet import CostNet
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import click
from tqdm import tqdm
from pathlib import Path

@click.command()
@click.option('--lr', default=1e-4, help='learning_rate')
@click.option('--epochs', default=500, help='epochs')
@click.option('--batch', default=128, help='batch')
@click.option('--system', default='sst_envs')
@click.option('--model', default='acrobot_obs')
@click.option('--setup', default='default')
def train(lr, epochs, batch, system, model, setup):
    mpnet = MPNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4)
    mpnet.load_state_dict(torch.load('output/acrobot_obs/{}/ep500'.format(setup)))
    
    
    costnet = CostNet(ae_input_size=32, ae_output_size=1024, in_channels=1, state_size=4, encoder=mpnet.encoder)
    
    net = costnet
    for param in costnet.encoder.parameters():
        param.requires_grad = False
    
    env_vox = torch.from_numpy(np.load('{}/{}_env_vox.npy'.format(system, model))).float()
    if torch.cuda.is_available():
        net = net.cuda()
        env_vox = env_vox.cuda()
    #optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    train_loader, test_loader = get_loader_cost(system, model, batch_size=batch, setup=setup)
    for i in tqdm(range(epochs+1)):
        print("epoch {}".format(i))
        train_loss = []
        net.train()
        for data, label in tqdm(train_loader):
            ## prepare data
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            inputs = data[:,1:]
            envs = env_vox[(data[:, 0]).long()]
            ## execute
            optimizer.zero_grad()
            output = net(inputs, envs)
            loss = F.mse_loss(output, label)
            loss.backward()
            # print(loss.item())
            train_loss.append(loss.item())
            optimizer.step()
            # break
        #scheduler.step(i)
        print("train loss:{}".format(np.mean(train_loss)))
        for data, label in test_loader:
            net.eval()
            eval_loss = []
            with torch.no_grad():
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()                
                inputs = data[:,1:]
                envs = env_vox[(data[:,0]).long()]
                output = net(inputs, envs)
                loss = F.mse_loss(output, label)
                eval_loss.append(loss.item())
        print("eval loss:{}".format(np.mean(eval_loss)))
        if i%1 == 0:
            Path("output/{}/{}/costnet".format(model, setup)).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), "output/{}/{}/costnet/ep{}".format(model, setup, i))


if __name__ == "__main__":
    train()
