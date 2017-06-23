# OpenCV-Python
1. [Instalaci&oacute;n](#instalacion)
    1. [Windows](#windows)
        1. [TensorFlow](#win-tensorflow)
        2. [OpenCV](#win-opencv)
    2. [Linux](#linux)
	    1. [TensorFlow](#lx-tensorflow)
		2. [OpenCV](#lx-opencv)
2. [Uso de los programas](#uso-de-los-programas)
    1. [Guantes](#guantes)
    2. [Buffy](#buffy)
    3. [FaceInfo](#faceinfo)
    4. [TF](#tf)
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

#### <a name="win-tensorflow"></a> TensorFlow
Para la instalacion de TensorFlow se utilizaran los siguientes comandos:
- Para la versi&oacute;n sin GPU:
```sh
(tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.2.0-cp35-cp35m-win_amd64.whl
```
- Para la versi&oacute;n con GPU:
```sh
(tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.0-cp35-cp35m-win_amd64.whl
```
Recordar que para la versi&oacute;n con GPU se deben instalar antes [CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey) y los drivers correspondientes.

#### <a name="win-opencv"></a> OpenCV
Para la instalaci&oacute;n de OpenCV en Windows de 64 bits con Python 3.5 se recomienda descargarlo del [siguiente enlace](http://www.lfd.uci.edu/~gohlke/pythonlibs/vu0h7y4r/opencv_python-3.2.0+contrib-cp35-cp35m-win_amd64.whl) o buscar la versi&oacute;n correspondiente en [esta p&aacute;gina](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
Luego se instalar&aacute; con el comando pip el archivo .whl descargado:
```sh
(tensorflow)C:\Downloads> pip install opencv_python-3.2.0+contrib-cp35-cp35m-win_amd64.whl
```

### <a name="linux"></a> Linux
Para las distribuciones de Linux se utiliza la creaci&oacute;n de entornos virtuales dentro de Python para separar los proyectos que necesiten diferentes versiones del int&eacute;rprete o de las librer&iacute;as que se utilicen.  

Para este caso en el que utilizamos Python 3 se realizan los siguientes pasos:
1. Instalar pip3 y virtualenv, esto instalar&aacute; Python 3 si no se encuentra en el sistema.
```sh
$ sudo apt-get install python3-pip python3-dev python-virtualenv
```
2. Crear el entorno con virtaulenv, el flag `system-site-packages` se utiliza para tomar las librer&iacute;as que ya est&aacute;n instaladas en el sistema globalmente, ignorarlo en caso de no querer incluirlas. `ENV_DIRECTORY` es el directorio en el que se crear&aacute; el entorno.
```sh
$ virtualenv --system-site-packages -p python3 <ENV_DIRECTORY>
```
3. Activar el entorno virtualenv creado.
```sh
$ source <ENV_DIRECTORY>/bin/activate
```
4. Para salir del entorno se debe ejecutar:
```sh
(<ENV_NAME>)$ deactivate
```

#### <a name="lx-tensorflow"></a> TensorFlow
Tenemos dos formas de instalar esta librer&iacute;a: desde el c&oacute;digo fuente o desde pip.
Si se desea instalar el soporte para GPU se deber&aacute; instalar [CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey) y los drivers correspondientes antes de seguir.  
Tambi&eacute;n es necesaria la librer&iacute;a CUDA Profile Tools Interface:
```sh
$ sudo apt-get install libcupti-dev
```
##### Pip
Para la instalaci&oacute;n con este m&eacute;todo se debe ejecutar dentro del entorno:
```sh
(<ENV_NAME>)$ pip3 install --upgrade tensorflow       # Tensorflow
(<ENV_NAME>)$ pip3 install --upgrade tensorflow-gpu   # Tensorflow y GPU
```
##### Desde el c&oacute;digo fuente
Este m&eacute;todo de instalaci&oacute;n es el recomendado ya que es el que mejor resultados de performance da al correr los programas que utilicen esta libreri&iacute;a.  
Primero hay que instalar algunas dependencias:
- Bazel, para esto seguir las [siguientes instrucciones](https://bazel.build/versions/master/docs/install-ubuntu.html)
- Dependencias de Python de TensorFlow (`numpy`,`wheel`) dentro del entorno.
- (Opcional) Librer&iacute;as para el soporte de la GPU descritas anteriormente.

Luego se deben realizar los siguientes pasos:
1. Clonar el repositorio de TensorFlow y elegir la versi&oacute;n a compilar entre todos los branches.
```sh
$ git clone https://github.com/tensorflow/tensorflow
$ cd tensorflow
$ git checkout <BRANCH>   # Por ejemplo: r1.0
```
2. Luego se debe ejecutar el script de configuraci&oacute;n, en el cual se nos pedir&aacute; ingresar varias opciones para el uso de TensorFlow. Por ejemplo, directorio de Python, soporte para Google Cloud Platform, Hadoop, flags de optimizaci&oacute;n. Utilizar los directorios del entorno virtual (*A confirmar*).
```sh
$ ./configure
```

3. Luego se ejecuta el siguiente comando.
```sh
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package  # GPU
```
> NOTE on gcc 5 or later: the binary pip packages available on the TensorFlow website are built with gcc 4, which uses the older ABI. To make your build compatible with the older ABI, you need to add --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" to your bazel build command. ABI compatibility allows custom ops built against the TensorFlow pip package to continue to work against your built package.

>Tip: By default, building TensorFlow from sources consumes a lot of RAM. If RAM is an issue on your system, you may limit RAM usage by specifying --local_resources 2048,.5,1.0 while invoking bazel.

4. Esto generar&aacute; un script que se debe ejecutar para crear el archivo .whl
```sh
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

5. Finalmente, dentro del entorno se instala el archivo .whl con pip.
```sh
(<ENV_NAME>)$ pip3 install /tmp/tensorflow_pkg/<WHL_FILE>
```

#### <a name="lx-opencv"></a> OpenCV
Para Linux no se dispone de un archivo .whl como en Windows. La opci&oacute;n m&aacute;s recomendada es compilar OpenCV desde su c&oacute;digo fuente para la versi&oacute;n de Python de nuestro entorno virtual.
1. Primero instalamos algunas dependencias dentro del sistema:
```sh
$ sudo apt-get install build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran
```

2. Luego descargamos la &uacute;ltima versi&oacute;n del c&oacute;digo fuente de OpenCV de su repositorio oficial junto con el repositorio `opencv_contrib` para tener una instalaci&oacute;n completa de la herramienta y no s&oacute;lo las funciones b&aacute;sicas.
```sh
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip
$ unzip opencv.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip
$ unzip opencv_contrib.zip
```

3. Dentro del entorno instalamos `numpy` si todav&iacute;a no lo hicimos.
```sh
(<ENV_NAME>)$ pip3 install numpy
```

4. Luego compilamos OpenCV.
```sh
(<ENV_NAME>)$ cd opencv-3.2.0
(<ENV_NAME>)$ mkdir build
(<ENV_NAME>)$ cd build
(<ENV_NAME>)$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
                    -D CMAKE_INSTALL_PREFIX=<ENV_DIRECTORY>/local \
                    -D INSTALL_PYTHON_EXAMPLES=ON \
                    -D INSTALL_C_EXAMPLES=OFF \
                    -D OPENCV_EXTRA_MODULES_PATH=<OPENCV_CONTRIB_DIR>/modules \
                    -D PYTHON_EXECUTABLE=<ENV_DIRECTORY>/bin/python3 \
                    -D BUILD_EXAMPLES=ON ..
(<ENV_NAME>)$ make -j4
```

5. Procedemos a la instalaci&oacute;n con:
```sh
(<ENV_NAME>)$ make install
```


----
## <a name="uso-de-los-programas"></a> Uso de los programas
A continuaci&oacute;n se detalla la utilizaci&oacute;n de cada programa del repositorio, se asume que el ususario se encuentra en el entorno de Python correspondiente tanto en Windows como en Linux.

### <a name="guantes"></a> Guantes
Este fue el primer programa para realizar pruebas sobre un dataset de lenguaje de señas con guantes de colores bien definidos e identificables dentro de la imagen. En el procesamiento se segmenta los guantes con filtrado de colores, se busca los contornos de los guantes, adem&aacute;s de realizar recuadros para ubicar la posici&oacute;n de los mismos. Como una prueba adicional se utliza una cascada Haar para la detecci&oacute;n de la cara que utiliza el m&eacute;todo de Viola-Jones.
```sh
python imagenes.py -i <IMAGE>
```
### <a name="buffy"></a> Buffy
El dataset [Buffy Stickmen](http://www.robots.ox.ac.uk/~vgg/data/stickmen/) presenta imagenes del show televisivo *Buffy the Vampire Slayer* junto con anotaciones sobre las poses de las personas que aparecen en ellas. Su objetivo es proporcionar una base de datos de im&aacute;genes que no tengan restricciones y presente escenarios reales en los cuales utilizar diferentes algoritmos para la estimaci&oacute;n de poses.  
En este programa se realiza una prueba sobre el c&aacute;lculo de la precisi&oacute;n de la t&eacute;cnica de Viola-Jones para la detecci&oacute;n de la cara comparado con las anotaciones del dataset. Dado que la detecci&oacute;n de la cara nos da un recuadro de la cara y las anotaciones presentan la l&iacute;nea del eje de la cabeza, esto afectar&aacute; en cierta medida los resultados obtenidos. Como medida de comparaci&oacute;n se utiliza la ra&iacute;z cuadrada de la distancia eucl&iacute;dea entre el punto central del recuadro de la cara y el punto medio entre los extremos de la l&iacute;nea anotada. Para la precisi&oacute;n de este m&eacute;todo se toma una distancia de referencia hasta donde la estimaci&oacute;n se considera correcta.  
Luego se genera un histograma de las distancias para visualizar la informaci&oacute;n. El programa adem&aacute;s cuenta con un modo debug que para cada imagen muestra los puntos comparados.
```sh
python buffy.py -p <BUFFY_PATH> [-d [DEBUG]]
```

### <a name="faceinfo"></a> FaceInfo
Este programa es una prueba para verificar los resultados de intentar detectar manos a partir del color de piel de la cara. Primero se utliza el m&eacute;todo de Viola-Jones para encontrar las caras en la imagen. Adem&aacute;s se calcula un &aacute;rea de inter&eacute;s en donde pueden estar las manos en base a la cara detectada. Luego con la t&eacute;cnica de Back Projection se extrae de la imagen todos los pixels con colores que coincidan a los de la cara. En base a ello se buscan algunos contornos que podr&iacute;n ser manos.
```sh
python faceinfo.py -i <IMAGE>
```
### <a name="tf"></a> TF
En esta secci&oacute;n se describir&aacute;n los programas que hagan uso de la librer&iacute;a TensorFlow.

#### Transferencia de aprendizaje
Esta es una t&eacute;cnica que consiste en la utilizaci&oacute;n de una red neuronal pre-entrenada para conseguir resultados intermedios de la misma y as&iacute; ahorrar el entrenamiento de una red compleja.  
Las pruebas realizadas fueron tomando la red convolucional [Inception-v3](https://arxiv.org/abs/1512.00567), entrenada y desarrollada por Google con datos del 2012 para la competici&oacute;n ImageNet logrando una tasa de error de predicci&oacute;n en el top 5 de sus resultados del 3.46% superando la capacidad de algunas personas como se puede apreciar en [este blog](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/).  
El funcionamiento de esta t&eacute;cnica es muy simple y el flujo de trabajo se podr&iacute;a definir de la siguiente forma:  

Pre-procesamiento de la imagen --> Imagen JPEG --> Inception-v3 --> Array[2048] --> Clasificaci&oacute;n --> Resultado  

Dado que se dispone de varios datasets con im&aacute;genes PNG de gestos segmentados y ya procesadas, s&oacute;lo hace falta la conversi&oacute;n al formato JPEG para ser procesadas por la red Inception-v3. De esto se encarga el programa `extract_features.py`, dado un dataset, genera los archivos JPEG temporalmente para ser procesados, y guarda en los archivos `features.pkl` y `labels.pkl` los arreglos obtenidos y la clase del gesto partiendo del nombre de la imagen. El modo de utilizaci&oacute;n de este programa es:
```sh
python extract_features.py -i <DATASET_DIRECTORY>
```
Luego, una vez obtenidos los arreglos de la salida de la red se pueden utilizar varios m&eacute;todos para su clasificaci&oacute;n.  
Hasta la fecha (22/06/2017) se dispone de dos programas diferentes:
- Una que entrena una SVM simple, la cual parece funcionar bastante bien con los datasets dispuestos. 
```sh
python train_svm.py -f <FEATURES_FILE> -l <LABELS_FILE>
```
- Una red neuronal profunda fully-connected en la que quedan pendientes varios tests.
```sh
python train_nn.py -f <FEATURES_FILE> -l <LABELS_FILE>
```
  
Como resultado final se obtiene un proceso de clasificaci&oacute;n de gestos que aprovecha el entrenamiento de una red conocida y ahorrando pasos intermedios.


----
## <a name="resultados"></a> Resultados
TODO
## <a name="referencias"></a> Referencias
TODO