from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import constants
import os
from torch import optim
from cnn_autoencoder import CnnAutoencoder
from mlp_autoencoder import MLPAutoencoder
from helper import save_images


def main(device, train_data_loader, test_data_loader):
    # select which autoencoder to train
    # available autoencoders [MLP, CNN]
    is_mlp = False
    is_cnn = False
    if constants.RUN_CNN_AUTOEN:
        autoencoder = CnnAutoencoder()
        train_output_dir = constants.CNN_AUTO_EN_TRAIN_OUT_DIR
        test_output_dir = constants.CNN_AUTO_EN_TEST_OUT_DIR
        model_path = constants.CNN_AUTO_EN_MODEL_PATH
        is_cnn = True
    elif constants.RUN_MLP_AUTOEN:
        is_mlp = True
        autoencoder = MLPAutoencoder()
        train_output_dir = constants.MLP_AUTO_EN_TRAIN_OUT_DIR
        test_output_dir = constants.MLP_AUTO_EN_TEST_OUT_DIR
        model_path = constants.MLP_AUTO_EN_MODEL_PATH

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
            images = images.to(device)
            if is_mlp:
                images = images.view(-1, 28 * 28)
            output = autoencoder(images)
            loss = criterian(output, images.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_data_loader)
        if epoch % constants.PRINT_EVERY == 0:
            print(
                f"epoch {epoch}/{constants.NUM_EPOCHS} loss is :{train_loss}")
            save_images(images[0], train_output_dir, epoch)
            save_images(output[0], train_output_dir, epoch, 'out')
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
            if is_mlp:
                images = images.view(-1, 28 * 28)
            output = autoencoder(images)
        for i in range(len(images)):
            save_images(images[i], test_output_dir, i)
            save_images(output[i], test_output_dir, i, image_name='out')
    print("Process completed!")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = constants.DATA_DIR
    if not os.path.exists(constants.DATA_DIR):
        os.mkdir(data_dir)

    # apply torch transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
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
