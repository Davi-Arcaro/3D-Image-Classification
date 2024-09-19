#Seguindo a página https://keras.io/examples/vision/3D_image_classification/
import os
import zipfile
import numpy as np
import tensorflow as tf #para processamento de dados
import keras
from keras import layers
import nibabel as nib #Necessário para ler as imagens do banco de dados, estas estão em formato .nii
from scipy import ndimage
import random
import matplotlib.pyplot as plt

os.chdir('C:/Users/fanta/OneDrive/Documentos/Programming/PythonProjects/TreinodeProgramação/InteligenciaArtificial')
#A base de dados será uma tomografia computadorizada (CT) de pulmões com e sem COVID

#Base de dados de tomografias computadorizadas anormais
url_ct0 = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename_ct0 = os.path.join(os.getcwd(), "CT-0.zip")
keras.utils.get_file(filename_ct0, url_ct0)

#Base de dados de tomografias computadorizadas normais
url_ct23 = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename_ct23 = os.path.join(os.getcwd(), "CT-23.zip")
keras.utils.get_file(filename_ct23, url_ct23)

#Fazendo um diretório para guardar os dados
os.makedirs("MosMedData")

#Unzipando os dados no novo diretório
with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

def read_nifti_file(filepath):
    # Ler e carregar o volume
    #Ler o arquivo
    scan = nib.load(filepath)
    #Pegar os dados crus
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    #Normalizando o volume
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype('float32')
    return volume

#Nessa função resize_volume, 3 processamentos cruciais serão definidos:
#1. Rotacionar as imagens em 90° para que a orientação delas esteja fixa;
#2. Escalonar os valores de HU (Houndsfield Units, são unidades de medidas das CTs)
#3. Resize da largura, altura e profundidade
def resize_volume(img):
    #Resize no eixo z
    #Setando a profundidade desejada
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    #Pegando a profundidade atual
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    #Computando o fator profundidade
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    heigth_factor = 1 / height
    #Rotacionando
    img = ndimage.rotate(img, 90, reshape=False)
    #Resize pelo eixo z
    img = ndimage.zoom(img, (width_factor, heigth_factor, depth_factor), order=1)
    return img

def process_scan(path):
    #Ler e dar um resize no volume
    #Ler o scan
    volume = read_nifti_file(path)
    #Normalizar
    volume = normalize(volume)
    #Resize da largura, altura e profundidade
    volume = resize_volume(volume)
    return volume

#Ler os caminhos dos scans CTs dos diretórios classe
#Arquivo "CT-0" consiste de scans em que o tecido pulmonar está normal, nenhum sinal de pneumonia viral
normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
#Arquivo "CT-23" consiste em scans CTs tendo várias opacificações em vidro fosco, envolvimento de parênquima pulmonar
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-23", x)
    for x in os.listdir("MosMedData/CT-23")
]

print(f'Scanners CT contendo tecido pulmonar normal: {str(len(normal_scan_paths))}')
print(f'Scanners CT contendo tecido pulmonar anormal: {str(len(abnormal_scan_paths))}')

#Construindo datasets de treinamento e validation

#Ler e processar os scans
#Cada scan é redimensionado(resize) a altura, largura e profundidade. Todos são redimesionados
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

#Para a presença de scans CTs com a presença de virose pulmonar
#Designar 1, para os normais designar 0
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

#Dividindo os dados na proporção 70-30 para treinamento e validação
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
    "Número de amostras no treinamento e validação são %d e %d."
    % (x_train.shape[0], x_val.shape[0])
)

#Data Augmentation
#Scans CT são augmentados no treinamento. Já que os dados são guardados em tensores rank-3 de shape(samples, height, width, depth), adicionamos uma dimensão de 1 no axis 4 para poder fazer convoluções em 3D nos dados.
#O novo shape é (samples, height, width, depth, 1). Existem vários tipos de preprocessamento e técnicas de augmentation. Nesse exemplo, são mostrados alguns simples para começar

def rotate(volume):
    #Rotacionar o volume em alguns graus
    def scipy_rotate(volume):
        #Definir alguns ângulos de rotação
        angles = [-20, -10, -5, 5, 10, 20]
        #Selecionar ângulos aleatórios
        angle = random.choice(angles)
        #Rotacionar o volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume
    
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    #Processar os dados de treinamento rotacionando eles e adicionando um canal
    #Rotacionar o volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    #Processar a validation data adicionando apenas um canal
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

#Ao definir o train e validation data loader, os dados de treinamento são passados e a função de augmentação randomicamente rotaciona o volume em diferentes ângulos.
#Note que âmbos os dados de treinamento e validação já estão remodelados(reshape) para terem valores entre 0 e 1

#Definindo data loaders
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training. (?)
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
#Apenas redimensionar
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
#Visualizando um CT scan augmentado:
data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimensão do CT scan é: ", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap='gray')

#CT scans possuem muitos slices, visualizando alguns slices
def plot_slices(num_rows, num_columns, width, height, data):
    #Plotando uma montagem de 20 slices
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights}
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap='gray')
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

#Visualizando uma montagem de slices.
#4 linhas e 10 colunas para 100 slices do CT scan
plot_slices(4, 10, 128, 128, image[:, :, :40])

#Definindo a Rede Neural Convolucional 3D
#Para ser mais fácil de compreender o modelo, ele será separado em blocos
def get_model(width=128, height=128, depth=64):
    #Construindo uma rede neural convolucional em 3D
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation='relu')(x)
    x= layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation='sigmoid')(x)

    #Definindo o modelo
    model = keras.Model(inputs, outputs, name='3dcnn')
    return model

#Construindo o modelo
model = get_model(width=128, height=128, depth=64)
model.summary()

#Compilando o modelo
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['acc'], run_eagerly=True)

#Definindo callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint('3d_image_classification.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_acc', patience=15)

#Treinando o modelo, fazendo a validação no final ed cada epoch
epochs = 100
model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, shuffle=True, verbose=2, callbacks=[checkpoint_cb, early_stopping_cb])

#Visualizando a performace do modelo
#Aqui a accuracy e loss para o treinamento e validação são plotados.
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(['acc', 'loss']):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history['val_' + metric])
    ax[i].set_title('Model {}'.format(metric))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(metric)
    ax[i].legend(['train', 'val'])

#Fazendo previsões m um único CT scan
#Carregando os melhores pesos
model.load_weights('3d_image_classification.keras')
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ['normal', 'abnormal']
for score, name in zip(scores, class_names):
    print('Este model é %.2f porcento confiante de o CT scan é %s' % ((100 * score), name))

