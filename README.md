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


