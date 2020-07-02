import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image


def plot_image(image):
    np_image = image.numpy()
    np_image = np_image.reshape((28, 28))
    plt.imshow(np_image, cmap='Greys_r')
    plt.show()


def save_images(image, output_dir, epoch, image_name='input'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image = image.cpu().data
    image = 0.5 * (image + 1)
    image = image.clamp(0, 1)
    image = image.view(1, 28, 28)
    save_image(
        image, os.path.join(output_dir,
                            f"{image_name}_image_epoch{epoch}.png"))


def save_images_denoiser(image, output_dir, epoch, image_name='input'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image = image.cpu().data
    image = image.view(1, 28, 28)
    save_image(
        image, os.path.join(output_dir,
                            f"{image_name}_image_epoch{epoch}.png"))
