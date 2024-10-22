

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
#windows
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#set KMP_DUPLICATE_LIB_OK=True
#linux
#export KMP_DUPLICATE_LIB_OK=True

# Cargar y transformar la imagen
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convertir a escala de grises
        transforms.ToTensor()    # Convertir a tensor
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Añadir un batch dimension
    return image

# Crear el filtro Gaussiano
def gaussian_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = x.repeat(size, 1)
    y = x.t()
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalizar el kernel
    return kernel.unsqueeze(0).unsqueeze(0)

# Aplicar el filtro Gaussiano
def apply_gaussian_filter(image, kernel):
    return F.conv2d(image, kernel, padding=kernel.size(-1) // 2)

# Parámetros del filtro Gaussiano
kernel_size = 7  # Tamaño del kernel
sigma = 1.0      # Desviación estándar

# Cargar imagen
image_path = 'fig2.jpg'
image = load_image(image_path)

# Crear y aplicar el filtro Gaussiano
gaussian_kernel = gaussian_kernel(kernel_size, sigma)
blurred_image = apply_gaussian_filter(image, gaussian_kernel)

#imprimir filtro
print(gaussian_kernel)

# Visualizar resultados
plt.figure(figsize=(15, 5))

# Mostrar imagen original
plt.subplot(1, 3, 1)
plt.title('Imagen Original')
plt.imshow(image.squeeze(), cmap='gray')

# Mostrar imagen desenfocada
plt.subplot(1, 3, 2)
plt.title('Imagen con Filtro Gaussiano')
plt.imshow(blurred_image.squeeze().detach().numpy(), cmap='gray')

# Gráfico meshgrid del kernel Gaussiano
fig = plt.subplot(1, 3, 3, projection='3d')
x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
y = x.clone()
X, Y = torch.meshgrid(x, y, indexing='xy')
Z = gaussian_kernel.squeeze()

ax = plt.axes(projection='3d')
ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis')
ax.set_title('Kernel Gaussiano')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Valor')
plt.show()



