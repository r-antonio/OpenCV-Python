# OpenCV-Python
1. [Instalaci&oacute;n](#instalacion)
    1. [Windows](#windows)
        1. [TensorFlow](#tensorflow)
        2. [OpenCV](#opencv)
    2. [Linux](#linux)
2. [Uso de los programas](#uso-de-los-programas)
    1. [Buffy](#buffy)
    2. [TensorFlow](#tf)
3. [Resultados](#resultados)
4. [Referencias](#referencias)
## <a name="instalacion"></a> Instalaci&oacute;n de las herramientas
### <a name="windows"></a> Windows
En Windows se recomienda el entorno Anaconda por la facilidad de instalaci&oacute;n de algunas dependencias de Python que de otra forma no se pod&iacute;an instalar, por ejemplo el paquete `scikit-learn`.
- [Anaconda Win64 (Python 3.6)](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe)

Luego se crea un entorno virtual con la versi&oacute;n de Python 3.5 para poder trabajar con TensorFlow
```sh
C:> conda create -n tensorflow python=3.5
```
Luego, el entorno se activa ejecutando
```sh
C:> activate tensorflow
```
Dentro del entorno ya se pueden instalar todas las dependencias de Python que sean necesarias para cada programa con el comando pip sin ning&uacute;n problema.

#### <a name="tensorflow"></a> TensorFlow
Para la instalacion de TensorFlow se utilizaran los siguientes comandos:
- Para la versi&oacute;n sin GPU:
```sh
(tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.2.0-cp35-cp35m-win_amd64.whl
```
- Para la versi&oacute;n con GPU:
```sh
(tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.0-cp35-cp35m-win_amd64.whl
```
Recordar que para la version con GPU se deben instalar antes [CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey) y los drivers correspondientes.

#### <a name="opencv"></a> OpenCV
Para la instalaci&oacute;n de OpenCV en Windows se recomienda descargarlo del [siguiente enlace](http://www.lfd.uci.edu/~gohlke/pythonlibs/vu0h7y4r/opencv_python-3.2.0+contrib-cp35-cp35m-win_amd64.whl).
Luego se instalar&aacute; con el comando pip el archivo whl:
```sh
(tensorflow)C:\Downloads> pip install opencv_python-3.2.0+contrib-cp35-cp35m-win_amd64.whl
```

### <a name="linux"></a> Linux
----
## <a name="uso-de-los-programas"></a> Uso de los programas
A continuaci&oacute;n se detalla la utilizaci&oacute;n de cada programa del repositorio

### <a name="buffy"></a> Buffy
TODO
### <a name="tf"></a> TensorFlow
En esta secci&oacute;n se describir&aacute;n los programas que hagan uso de la librer&iacute;a TensorFlow.

#### Transferencia de aprendizaje
Esta es una t&eacute;cnica que consiste en la utilizaci&oacute;n de una red neuronal pre-entrenada para conseguir resultados intermedios de la misma y as&iacute; ahorrar el entrenamiento de una red compleja.  
Las pruebas realizadas fueron tomando la red convolucional [Inception-v3](https://arxiv.org/abs/1512.00567), entrenada y desarrollada por Google con datos del 2012 para la competici&oacute;n ImageNet logrando una tasa de error de predicci&oacute;n en sus 5 primeros resultados del 3.46% superando la capacidad del ser humano.

## <a name="resultados"></a> Resultados
TODO
## <a name="referencias"></a> Referencias
TODO
