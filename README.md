# IA-de-deteccion-de-violencia
Este proyecto implementa una red neuronal convolucional 3D (3D-CNN) para detectar escenas violentas en una transmisión de video. El 3D-CNN es un enfoque de aprendizaje supervisado profundo que aprende características violentas espaciotemporales de videos (secuencia de cuadros de imagen). A diferencia de las circunvoluciones 2D, este enfoque opera núcleos 3D en una serie de marcos de imagen en su contexto, lo que produce mapas de activación 3D que capturan características tanto espaciales como temporales que no se pueden identificar correctamente con las circunvoluciones 2D.
## Etapas del sistema
![image](https://user-images.githubusercontent.com/54364070/177218033-1cfdcb29-167f-4af3-b232-6bbb5f09a1fe.png)
Fuente de la imagen: [1]

Cada etapa en la preparación del proyecto se muestra en la siguiente imagen:

* *Preprocesamiento:* Para evitar el exceso de trabajo del motor de inferencia, el modelo MobileNet preentrenado (del Intel OpenVINO Model Zoo) se utiliza para detectar personas en un marco determinado. Cuando se detecta a una persona, se recopila una pila de 16 fotogramas y se pasa por el modelo 3D-CNN para detectar violencia.
* *Detección de violencia:* Cada secuencia de 16 cuadros se pasa a través del modelo 3D CNN capacitado que genera si la escena es violenta o no como puntajes de probabilidad. La clase con salida máxima es el valor predicho.
* *Visualización:* Una interfaz front-end permite visualizar el funcionamiento del sistema en tiempo real. El cuadro de video se reproduce en la pantalla con un área indicadora que señala escenas violentas.

* *Alerta:* Cuando se detecta violencia en cualquiera de las escenas, se notifica al grupo de seguridad más cercano para una respuesta inmediata.
## Entrenamiento del modelo

El modelo 3D CNN se entrenó de forma personalizada utilizando la arquitectura que se muestra a continuación:

### Conjuntos de datos
Se combinaron tres bases de datos para esta tarea: [Violencia en deportes](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech),[Peliculas](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635) y [Violencia en espacios públicos](https://www.openu.ac.il/home/hassner/data/violentflows/).

La base de datos de Hockey Fight contenía 1000 videoclips, la mitad con escenas violentas y la otra no violenta. La base de datos de Películas contenía 200 videoclips, la mitad con escenas violentas y la otra no violenta. La base de datos de violencia de la multitud contenía 246 videoclips de YouTube, la mitad con escenas violentas y la otra no violenta. Esto dio un total de 1446 videos, con 723 videos de violencia y no violencia.

Los frames de imagen se extraen de estos videos usando el script en `/data/video2img.sh`(obtenido de [JJBOY](https://github.com/JJBOY/C3D-pytorch)) a una frecuencia de muestreo de 16 fotogramas por segundo. Este valor fue elegido arbitrariamente y es lo suficientemente bueno para empezar. Luego, los diferentes marcos de imagen se recopilan en pilas con 16 marcos por pila usando `/data/create_stacks.py` utilizando la información proporcionada en `/data/train.txt` y `/data/test.txt` que especifica el punto de partida de cada pila. Esto era necesario ya que las tramas se enviaban en secuencias superpuestas.

Todo el conjunto de base de datos se dividió en conjunto de entrenamiento y conjunto de prueba en una proporción de 3:1. Esto fue de acuerdo con el método en [1].Luego, ambos conjuntos se empaquetaron en el formato HDF5 usando el paquete `h5py`.

### Preprocesamiento de datos

Primero se cambia el tamaño de cada pila de imágenes a 128 por 171 píxeles antes de recortarlas a 112 por 112 píxeles de acuerdo con la forma de entrada de la CNN 3D. Luego se convierten a tensores PyTorch y cada uno de los marcos RGB se normaliza con `mean=[0.485, 0.456, 0.406]` y `std=[0.229, 0.224, 0.225]`. Esta es una práctica de transformación común que se deriva de la normalización introducida por ImageNet.

### Hiperparámetros y Optimización

Para la tarea de entrenamiento se utilizaron los siguientes hiperparámetros:

* `num_epochs = 100`
* `tamaño_lote = 30`
* `tasa_de_aprendizaje = 0.003`

Estos parámetros no son de ninguna manera óptimos, pero dieron un resultado bastante bueno para empezar.

El optimizador de descenso de gradiente estocástico se utilizó para el aprendizaje con la función Cross Entropy como criterio para la pérdida de clasificación.

### Configuración de entrenamiento

Los conjuntos de datos de prueba y entrenamiento se cargaron en Google Drive, ya que todo el entrenamiento se realizó en Google Colab. Colab es un servicio en la nube gratuito que brinda acceso a instancias de GPU gratuitas y es compatible con las bibliotecas más conocidas, incluida PyTorch, que se utilizó para este proyecto. El entorno de formación tiene las siguientes especificaciones:

* GPU: 1xTesla K80, cómputo 3.7, con 2496 núcleos CUDA, 12GB GDDR5 VRAM
* CPU: Procesadores Xeon de 1 núcleo único con hiperproceso a 2,3 Ghz, es decir (1 núcleo, 2 subprocesos)
* RAM: ~12,6 GB disponibles
* Disco: ~33 GB Disponible

## Resultados del modelo

### Pérdida de prueba

![image](https://user-images.githubusercontent.com/54364070/177221692-3aa31922-73f8-43b4-9330-32ca1041eca7.png)

### Exactitud de la prueba

![image](https://user-images.githubusercontent.com/54364070/177222202-bfc02b29-47a4-4c7a-a2ca-12b60a05f1fe.png)

La mejor precisión de 84% se obtuvo en el ciclo de entrenamiento 36.

Este modelo preliminar definitivamente está lejos de ser el mejor.El modelo estaba sujeto a ajustes excesivos y se pueden lograr muchas mejoras con el entrenamiento adecuado.

## Inferencia de borde

### El kit de herramientas Intel OpenVINO™

El kit de herramientas OpenVINO™ viene con dos herramientas especiales que hemos utilizado mucho en este proyecto,el _Model Optimizer (MO)_ y _Inference Engine (IE)_.El MO analiza el modelo de aprendizaje profundo sin procesar de los diferentes marcos (PyTorch en este caso) y se ajusta para una ejecución óptima en los dispositivos de borde.Técnicas como la congelación, la fusión y la cuantificación ayudan a reducir los tiempos de ejecución de la inferencia mientras mantienen la precisión en un umbral razonable. El MO genera un archivo XML generix que describe la arquitectura del modelo optimizado y un archivo BIN que contiene los pesos de cada capa en el modelo. El IE toma estos archivos y garantiza una inferencia acelerada en procesadores basados en Intel (como la instancia de AWS en la que se configuró este proyecto). El IE es originalmente una biblioteca de C++, pero expone una API que se usa para interactuar con él en Python como se usa en este proyecto. En su mayor parte, el código repetitivo proporcionado en el curso Intel Edge AI Fundamentals se utilizó para construir este proyecto.

### Parámetros modificables

Actualmente, el modelo acepta dos parámetros de la siguiente manera:

`-m model_path` = "La ruta al archivo XML del modelo"
`-v video_path` = "La ruta al archivo de video"

Sin embargo, la mayoría de estos se han ocultado detrás de la API de la aplicación web para que los usuarios hagan todo esto sin preocuparse por las complejidades. Dado que solo se ha proporcionado un modelo en la instancia de la nube, ese argumento no está expuesto al público, pero ya está configurado y simplemente usa el valor predeterminado. La ruta del video se envía al modelo a través de la llamada a la API `/getinference/video_path`. El servidor analiza esto y lo envía al modelo a través del argumento `-v video_path`. En el futuro, se pueden agregar argumentos para el complemento específico del dispositivo (CPU, GPU, FPGA, etc.) o el tipo de procesamiento (SYNC, ASYNC).

## Ejecutando la aplicación

Para probar el modelo, puede usar [el siguiente link para el aplicativo Weeb general](https://keen-lumiere-1599db.netlify.com/). 

![image](https://user-images.githubusercontent.com/54364070/177222757-a3f028c1-4bcb-44be-bea5-37f15df0da77.png)

Una vez en la página como se muestra arriba, haga clic en 'Ejecutar inferencia'. Esto cargará la página de inferencia que le proporcionará una lista de videos para elegir.

![image](https://user-images.githubusercontent.com/54364070/177222789-1f42e112-f9e4-40b9-a828-d7b5dd017328.png)

Haga clic en cualquiera de los enlaces seleccionados y los resultados del video se mostrarán con el indicador rojo para escenas violentas y el indicador verde para escenas pacíficas.

## Pasos adicionales

* Entrene el modelo para una mayor precisión mediante la búsqueda de los mejores hiperparámetros y optimizador

La precisión de la identificación de escenas violentas en el flujo de video de entrada se puede mejorar significativamente reelaborando la canalización de entrenamiento. La mayoría de los hiperparámetros utilizados fueron por regla general y se pueden revisar para buscar los mejores parámetros para un rendimiento óptimo al mismo tiempo que se considera la precisión.

* Implementar sistema de alimentación multicámara

Se pretende que este sistema se implemente en sistemas de cámaras de vigilancia distribuidos. Es posible un sistema de alimentación central que procese estos flujos de entrada manteniendo el rendimiento en términos de velocidad y confiabilidad. Ya se han revisado los recursos de Intel OpenVINO que cubren este tema y se pueden implementar en el futuro.

* Cree una aplicación web completa para la visualización

La aplicación web (para demostración) solo se ha construido de manera aproximada. Una buena parte de los componentes no se han conectado para encajar bien. Se pretende que la aplicación web se completa para el seguimiento eficaz del rendimiento de este sistema.

* Implementar geolocalización para mejorar el reporte de eventos de violencia

Otra característica importante que se puede considerar en el futuro será agregar geolocalización a cada instalación de cámaras de vigilancia con fines informativos.

## Referencias

[1] Ullah, F. U. M., Ullah, A., Muhammad, K., Haq, I. U., & Baik, S. W. (2019). Violence Detection Using Spatiotemporal Features with 3D Convolutional Neural Network. *Sensors*, 19(11), 2472. https://doi.org/10.3390/s19112472

[2] Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning Spatiotemporal Features with 3D Convolutional Networks. *2015 IEEE International Conference on Computer Vision (ICCV)*. https://doi.org/10.1109/iccv.2015.510


