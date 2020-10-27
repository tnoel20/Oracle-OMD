import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets, transforms
from oc_data_load import CIFAR10_Data

# Implement autoencoder with several different latent
# layer sizes. Train on CIFAR-10.

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # in_channels: 3, out_channels: 16, kernel_size: 3
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
            #nn.Conv2d(32, 64, 7)
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)
        
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()#Sigmoid()
        )
        
    def forward(self, x):
        x, indices = self.encoder(x)
        x = self.unpool(x, indices)
        x = self.decoder(x)
        return x

    def get_latent(self, x):
        return self.encoder(x)


def train(model, device, tr_data, val, num_epochs=5, learning_rate=1e-3):#batch_size=64 
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, 
                                 weight_decay=1e-5) # <--
    outputs = []
    epoch = 1
    # Initial conditions
    val_loss_list = [1E6, 0]
    EPS = 1e-3
    #for epoch in range(num_epochs):
    while abs(val_loss_list[epoch] - val_loss_list[epoch-1]) > EPS and epoch < num_epochs:
        for i,img_batch in enumerate(tr_data):
            img_batch = img_batch.to(device)
            recon = model(img_batch)
            loss = criterion(recon, img_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch, float(loss)))
        outputs.append((epoch, img_batch, recon),)
        _, val_loss = get_val_loss(model, val, device)
        val_loss_list.append(val_loss)
        epoch += 1
    return outputs

# TODO: Fix this...
def get_val_loss(model, val, device):
    criterion = nn.MSELoss()
    outputs = []
    loss = 0

    # TODO: Update for val. data_loader no longer works here!!!
    for img_batch in val:
        # Reshape mini-batch data to [N, 32*32] matrix
        # Load it to the active device
        # batch_features = batch_features.view(-1, 32*32).to(device)
        img_batch = img_batch.to(device)
        # compute reconstructions
        reconstruction = model(img_batch)

        # compute training reconstruction loss
        test_loss = criterion(reconstruction, img_batch)

        # add the mini-batch training loss to epoch loss
        loss += test_loss.item()

    # Compute the epoch training loss
    loss = loss / len(val)
    # display the epoch test loss
    print("dev loss = {:.6f}".format(loss))
    outputs.append((None,img_batch,reconstruction),)

    return outputs, loss
    

def training_progression(outputs):
    num_epochs = len(outputs)
    for k in range(0, num_epochs, 5):
        plt.figure(figsize=(9,2))
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy() #.cpu().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2,9,i+1)
            plt.imshow(item[0])
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2,9,9+i+1)
            plt.imshow(item[0])
    plt.show()


def get_vanilla_ae(tr=None, val=None, filename='plain_ae.pth'):
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 20

    tr_loader = torch.utils.data.DataLoader(tr, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=4, shuffle=True, pin_memory=True)

    if os.path.isfile(filename):
        # Load model
        model = Autoencoder()
        model.load_state_dict(torch.load(filename))
        model.eval()
    else:
        # Train model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Autoencoder().to(device)
        # train and val are dataloaders
        outputs = train(model, device, tr_loader, val_loader, num_epochs=NUM_EPOCHS)
        torch.save(model.state_dict(), filename)

    return model

'''
def main():
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 20

    cifar_tr_data = datasets.CIFAR10(
        'data', train=True, download=True, transform=transforms.ToTensor()
    )
    cifar_test_data = datasets.CIFAR10(
        'data', train=False, download=True, transform=transforms.ToTensor()
    )

    cifar_tr_data = list(cifar_tr_data)[:4096] 
    cifar_test_data = list(cifar_test_data)[:2048]
    
    model = Autoencoder()
    max_epochs = 20
    outputs = train(model, cifar_tr_data, num_epochs=max_epochs)

    training_progression(outputs)

    test_loader = torch.utils.data.DataLoader(cifar_test_data)

    original = None
    for data in test_loader:
        original, _ = data
        break

    latent = model.get_latent(original)
    original = np.swapaxes(np.squeeze(original.detach().numpy()),0,2)
    latent = np.swapaxes(np.squeeze(latent.detach().numpy()),0,2)
    
    plt.subplot(2,1,1)
    plt.imshow(original)

    plt.subplot(2,1,2)
    plt.imshow(latent)
    
    
    
    # PREVIOUS BOUNDARY
    # Use the GPU
    device = torch.device("cuda")

    hidden_dim = 64
    # Create an instance of Autoencoder
    model = Autoencoder(CIFAR10_DIM, hidden_dim).to(device)

    # Note that loss is MSE
    outputs, loss, optimizer = train_model(model, train_loader, device)
    test_outputs, test_loss = compute_test_loss(model, test_loader, device)

    training_progression(outputs)
    #training_progression(test_outputs)

    
    latent_model = Autoencoder(CIFAR10_DIM, hidden_dim)
    x = torch.randn(1,CIFAR10_DIM)
    enc_output = model.encoder_layer(x)

    plt.figure(figsize=(1,2))
    imgs = x
    recon = enc_output
    plt.subplot(2,1,1)
    plt.imshow(x)
    plt.subplot(2,1,2)
    plt.imshow(enc_output)

    plt.show()
    
    torch.save({'epoch': NUM_EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
               }, 'autoencoder_save.pth')
    
    
    ############################################################################
    l_model = Autoencoder(input_shape=CIFAR10_DIM, hid_units=hidden_dim) #.to(device)
    l_optimizer = optim.Adam(model.parameters(), lr=1e-3)

    checkpoint = torch.load('autoencoder_save.pth')
    l_model.load_state_dict(checkpoint['model_state_dict'])
    l_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    
    # TODO Create Encoder Class and Decoder classes, then save trained Encoder?
    # AND/OR tinker with state_dict to determine if a subset of layers can
    # be heldout or ignored

    
    ############################################################################

    hidden_units = [2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 32, 40, 45, 50, \
                    60, 64, 70, 80, 90, 100, 110, 120, 128]
    losses = []

    for hidden_dim in hidden_units:
        # Create an instance of Autoencoder
        model = Autoencoder(input_shape=CIFAR10_DIM, hid_units=hidden_dim).to(device)

        # Note that loss is MSE
        outputs, loss, _ = train_model(model, train_loader, device)
        test_outputs, test_loss = compute_test_loss(model, test_loader, device)

        #training_progression(outputs)

        losses.append(test_loss)

    for i in range(len(hidden_units)):
        print('For {} hidden units, test loss is {}'.format(hidden_units[i],losses[i]))

    sns.set(font_scale=1.5)
    plt.plot(hidden_units, losses)
    plt.xlabel('Hidden Units')
    plt.ylabel('Loss, MSE')
    plt.title('Autoencoder Reconstruction Error')
    plt.show()
    
    
if __name__ == '__main__':
    main()
'''
