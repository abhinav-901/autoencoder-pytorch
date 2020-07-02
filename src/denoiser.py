from torchvision.datasets import MNIST
import torch
import constants
import os
from torchvision import transforms
from torch import optim
from cnn_autoencoder import Denoiser
from helper import save_images_denoiser


def main(device, train_data_loader, test_data_loader):
    # select which autoencoder to train
    # available autoencoders [MLP, CNN]
    autoencoder = Denoiser()
    train_output_dir = constants.DENOISER_TRAIN_OUT_DIR
    test_output_dir = constants.DENOISER_TEST_OUT_DIR
    model_path = constants.DENOISER_MODEL_PATH

    # Migrate all operations on GPU if available else on CPU
    autoencoder.to(device)
    # Get the loss (MSE)
    criterian = autoencoder.criterian
    optimizer = optim.Adam(autoencoder.parameters(), constants.LR)
    train_loss = 0
    print("Training Initiated")
    for epoch in range(1, constants.NUM_EPOCHS + 1):
        for data in train_data_loader:
            images, _ = data
            optimizer.zero_grad()
            images = images.to(device)
            noisy_images = images + constants.NOISE_FACTOR * torch.randn(
                *images.shape).to(device)
            noisy_images = noisy_images.clamp(0., 1.)
            output = autoencoder(noisy_images)
            loss = criterian(output, images.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_data_loader)
        if epoch % constants.PRINT_EVERY == 0:
            print(
                f"epoch {epoch}/{constants.NUM_EPOCHS} loss is :{train_loss}")
            save_images_denoiser(noisy_images[0], train_output_dir, epoch)
            save_images_denoiser(output[0], train_output_dir, epoch, 'out')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print(f"saving checkpoint for epoch: {epoch}")
            torch.save(autoencoder.state_dict(),
                       os.path.join(model_path, f"epoch{epoch}.pth"))
        if epoch == constants.SAVE_CHECKPOINT:
            break

    # activating evaluation mode so better to switch off the autograd
    autoencoder.eval()
    with torch.no_grad():
        print("Performing inference on test set")
        for data in test_data_loader:
            images, _ = data
            images = images.to(device)
            noisy_images = images + constants.NOISE_FACTOR * torch.randn(
                *images.shape).to(device)
            noisy_images = noisy_images.clamp(0., 1.)
            output = autoencoder(noisy_images)

        for i in range(len(images)):
            save_images_denoiser(noisy_images[i], test_output_dir, i)
            save_images_denoiser(output[i],
                                 test_output_dir,
                                 i,
                                 image_name='out')
    print("Process completed!")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = constants.DATA_DIR
    if not os.path.exists(constants.DATA_DIR):
        os.mkdir(data_dir)

    # apply torch transformation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root=data_dir,
                          download=True,
                          train=True,
                          transform=transform)
    test_dataset = MNIST(root=data_dir,
                         download=True,
                         train=False,
                         transform=transform)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)

    # call main function
    main(device, train_data_loader, test_data_loader)
