# Ensambles Paralelos 

## Homogéneos [Ampliar Información](https://github.com/ale-uy/Ensemble-Methods/blob/main/Homogeneous_Parallel_Ensembles.ipynb)

* Los conjuntos (o ensambles) homogéneos paralelos promueven la diversidad de conjuntos a través de la aleatorización: muestreo aleatorio de ejemplos de entrenamiento y de características, o incluso la introducción de la aleatorización en el algoritmo de aprendizaje base.
* El embolsado (**bagging**) es un método de conjunto simple que se basa en (1) muestreo de arranque (o muestreo con reemplazo), mas conocido como **bootstrap sampling**, para generar diversas réplicas del conjunto de datos y entrenar diversos modelos, y (2) la agregación de modelos para producir una predicción de conjunto a partir de un conjunto de predicciones de aprendizaje base individuales.
* El embolsado y sus variantes funcionan mejor con estimadores inestables (árboles de decisión no podados, máquinas de vectores de soporte [SVM], redes neuronales profundas, etc.), que son modelos de mayor complejidad y/o no linealidad.
* **Random Forest** se refiere a una variante de embolsado diseñada específicamente para usar árboles de decisión aleatorios como estudiantes base. El aumento de la aleatoriedad aumenta considerablemente la diversidad del conjunto, lo que permite que el conjunto disminuya la variabilidad y suavice las predicciones.
* Pegar (**pasting**), es una variante de empaquetar, muestra ejemplos de entrenamiento sin reemplazo y puede ser efectivo en conjuntos de datos con una gran cantidad de ejemplos de entrenamiento.
* Otras variantes de empaquetamiento, como los subespacios aleatorios [*random subspaces*] (características de muestreo) y los parches aleatorios [*random patches*] (muestreo de características y ejemplos de entrenamiento), pueden ser efectivos en conjuntos de datos con alta dimensionalidad.
* **Extra Trees** es otro método de conjunto similar a una bolsa que está diseñado específicamente para usar árboles extremadamente aleatorios como estudiantes base. Sin embargo, Extra Trees no utiliza muestreo *bootstrap* ya que la aleatorización adicional ayuda a generar diversidad de conjuntos.
* Los bosques aleatorios proporcionan importancias de características para clasificar las más importantes desde un punto de vista predictivo.

## Heterogéneos [Ampliar Información](https://github.com/ale-uy/Ensemble-Methods/blob/main/Heterogeneous_Parallel_Ensembles.ipynb)

* Los métodos de conjuntos heterogéneos promueven la diversidad de conjuntos a través de la heterogeneidad; es decir, utilizan diferentes algoritmos de aprendizaje base para entrenar a los estimadores base.
* Los métodos de ponderación (weighting methods) asignan a las predicciones individuales del estimador base un peso que corresponde a su desempeño; a los mejores estimadores base se les asignan pesos más altos e influyen más en la predicción final general.
* Los métodos de ponderación utilizan una función de combinación predefinida para combinar las predicciones ponderadas de los estimadores de base individuales. Las funciones de combinación lineal (p. ej., suma ponderada) suelen ser eficaces y fáciles de interpretar. También se pueden utilizar funciones de combinación no lineales, aunque la complejidad añadida puede dar lugar a un sobreajuste (overfitting).
* Los métodos de metaaprendizaje (meta-learning methods) aprenden una función de combinación de los datos, en contraste con los métodos de ponderación, en los que tenemos que crear uno nosotros mismos.
* Los métodos de metaaprendizaje crean múltiples capas de estimadores. El método de metaaprendizaje más común es el apilamiento (**Stacking**), llamado así porque literalmente apila algoritmos de aprendizaje en un esquema de aprendizaje similar a una pirámide.
* El apilamiento simple crea dos niveles de estimadores. Los estimadores base se entrenan en el primer nivel y sus resultados se utilizan para entrenar un estimador de segundo nivel llamado *metaestimador*. Son posibles modelos de apilamiento más complejos con muchos más niveles de estimadores.
* El apilamiento a menudo puede sobreajustarse, especialmente en presencia de datos ruidosos. Para evitar el sobreajuste, el apilamiento se combina con la validación cruzada (CV) para garantizar que diferentes estimadores base vean diferentes subconjuntos del conjunto original de datos para una mayor diversidad.
* El apilamiento con CV, aunque reduce el sobreajuste, también puede ser computacionalmente intensivo, lo que lleva a largos tiempos de entrenamiento. Para acelerar el entrenamiento y protegerse contra el sobreajuste, se puede usar un solo conjunto de validación. Este procedimiento se conoce como mezcla (*blending*).
* Cualquier algoritmo de aprendizaje automático se puede utilizar como metaestimador en el apilamiento. La *regresión logística* es la más común y conduce a modelos lineales. Los modelos no lineales, obviamente, tienen mayor poder representativo, pero también tienen un mayor riesgo de sobreajuste.
* Tanto los enfoques de ponderación como los de metaaprendizaje pueden usar las predicciones del estimador base directamente o las probabilidades de predicción. Este último generalmente conduce a un modelo más suave y matizado.

# Ensambles Secuenciales
## Refuerzo Adaptativo [Ampliar Información](https://github.com/ale-uy/Ensemble-Methods/blob/main/Sequential_Ensembles_Adaptative_Boosting.ipynb)

* El refuerzo adaptativo (**AdaBoost**) es un algoritmo de conjunto secuencial que utiliza aprendices débiles como estimadores básicos.
* En la clasificación, un aprendiz débil es un modelo simple que funciona solo un poco mejor que adivinar al azar, es decir, 50% de precisión. Los tocones (stumps) de decisión y los árboles de decisión poco profundos son ejemplos de estudiantes débiles.
* AdaBoost mantiene y actualiza pesos sobre ejemplos de entrenamiento. Utiliza la reponderación tanto para priorizar ejemplos mal clasificados como para promover la diversidad de conjuntos.
* AdaBoost también es un conjunto aditivo en el sentido de que realiza predicciones finales a través de combinaciones (lineales) aditivas ponderadas de las predicciones de sus estimadores base.
* AdaBoost es generalmente resistente al sobreajuste, ya que reúne a varios alumnos débiles. Sin embargo, AdaBoost es sensible a los valores atípicos (*outliers*) debido a su estrategia de reponderación adaptativa, que aumenta repetidamente el peso de los valores atípicos durante las iteraciones.
* El rendimiento de AdaBoost se puede mejorar encontrando una buena compensación entre la tasa de aprendizaje (*learning rate*) y la cantidad de estimadores básicos.
* La validación cruzada con la búsqueda en cuadrícula (**grid search**) se implementa comúnmente para identificar el mejor equilibrio de parámetros entre la tasa de aprendizaje y la cantidad de estimadores.
* Debajo del capó, AdaBoost optimiza la función de pérdida exponencial (*exponential loss function*).
* **LogitBoost** es otro algoritmo de impulso que optimiza la función de pérdida logística. Se diferencia de AdaBoost en dos formas: (1) trabaja con probabilidades de predicción y (2) usa cualquier algoritmo de clasificación como algoritmo de aprendizaje base.

## Aumento de Gradiente [Ampliar Información](https://github.com/ale-uy/Ensemble-Methods/blob/main/Sequential_Ensembles_Gradient_Boosting.ipynb)

* El descenso de gradiente (*gradient descent*) se usa a menudo para minimizar una función de pérdida para entrenar un modelo de aprendizaje automático.
* Los residuos, o errores entre las etiquetas verdaderas y las predicciones del modelo, se pueden usar para caracterizar ejemplos de entrenamiento clasificados correctamente y mal clasificados. Esto es análogo a cómo AdaBoost usa pesos.
* El impulso de gradiente (**gradient boosting**) combina el descenso de gradiente y el impulso para aprender un conjunto secuencial de alumnos débiles.
* Los alumnos débiles en el aumento de gradiente son árboles de regresión que se entrenan sobre los residuos de los ejemplos de entrenamiento y aproximan el gradiente.
* El aumento de gradiente se puede aplicar a una amplia variedad de funciones de pérdida que surgen de tareas de clasificación o regresión.
* El aprendizaje de árboles basado en histogramas (**Histogram-based tree**) cambia la exactitud y la eficiencia, lo que nos permite entrenar modelos que aumentan el gradiente muy rápidamente y escalar a conjuntos de datos más grandes.
* El aprendizaje se puede acelerar aún más mediante el muestreo inteligente de ejemplos de capacitación (muestreo de un lado basado en gradientes, *GOSS*) o la agrupación inteligente de funciones (agrupación de funciones exclusivas, *EFB*).
* **LightGBM** es un potente marco disponible públicamente para potenciar gradientes que incorpora tanto GOSS como EFB.
* Al igual que con AdaBoost, podemos evitar el sobreajuste en el aumento de gradiente eligiendo una tasa de aprendizaje efectiva o mediante una parada temprana (*early stopping*). LightGBM proporciona soporte para ambos.
* Además de una amplia variedad de funciones de pérdida para regresión y clasificación, LightGBM también brinda soporte para la incorporación de nuestras propias funciones de pérdida personalizadas y específicas del problema para el entrenamiento.

## Impulso de Newton [Ampliar Información](https://github.com/ale-uy/Ensemble-Methods/blob/main/Sequential_Ensembles_Newton_boosting.ipynb)

* El descenso de Newton es otro algoritmo de optimización, similar al descenso de gradiente.
* El descenso de Newton usa información de segundo orden (*hessiana*) para acelerar la optimización en comparación con el descenso de gradiente, que solo usa información de primer orden (*gradiente*).
* El impulso de Newton (**Newton boosting**) combina el descenso y el impulso de Newton para entrenar un conjunto secuencial de alumnos débiles.
* El impulso de Newton utiliza residuos ponderados para caracterizar ejemplos de entrenamiento correctamente clasificados y mal clasificados. Esto es análogo a cómo AdaBoost usa pesos y cómo el aumento de gradiente usa residuos.
* Los alumnos débiles en el impulso de Newton son árboles de regresión que se entrenan sobre los residuos ponderados de los ejemplos de entrenamiento y se aproximan al paso de Newton.
* Al igual que el aumento de gradiente, el aumento de Newton se puede aplicar a una amplia variedad de funciones de pérdida que surgen de tareas de regresión o clasificación.
* La optimización de una función de pérdida regularizada (*regularized loss function*) ayuda a controlar la complejidad de los alumnos débiles en el conjunto aprendido, previene el sobreajuste y mejora la generalización.
* **XGBoost** es un marco poderoso y disponible públicamente para el impulso de Newton basado en árboles que incorpora el impulso de Newton, la búsqueda eficiente de divisiones y el aprendizaje distribuido.
* XGBoost optimiza un objetivo de aprendizaje regularizado que consiste en la función de pérdida (para ajustar los datos) y dos funciones de regularización: *regularización L2* y *número de nodos hoja*.
* Al igual que con AdaBoost y el aumento de gradiente, podemos evitar el sobreajuste en el aumento de Newton eligiendo una tasa de aprendizaje efectiva o mediante una parada temprana. XGBoost es compatible con ambos.
* XGBoost implementa un algoritmo de búsqueda de división aproximado llamado bosquejo de cuantiles ponderados (*weighted quantile sketch*), que es similar a la búsqueda de división basada en histograma pero adaptado y optimizado para un impulso de Newton eficiente.
* Además de una amplia variedad de funciones de pérdida para clasificación, regresión y "rankeo", XGBoost también brinda soporte para la incorporación de nuestras propias funciones de pérdida personalizadas y específicas del problema para el entrenamiento.
