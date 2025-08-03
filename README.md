# Prueba Técnica – Candidato a Gerente en Inteligencia Artificial e Innovación
**Objetivo:** Pronosticar si un cliente suscribirá un depósito a plazo **(target: y)** usando el dataset `bank-additional-full.csv` y así priorizar a los clientes más propensos a convertir.

## 1. Primeras observaciones después de leer `data_description.txt`

- La columna duration no debe ser usada para un modelo realista, ya que es conocida sólo después de la llamada.
- Las variables con valor *"unknown"* deben ser tratadas como missing **(posible imputación o categoría aparte)**.
- La variable objetivo y es binaria: *"yes"* o *"no"*

## 2. Punto de partida
La fuente proporcionada brinda la siguiente información que evita una resolución desde cero de la prueba técnica. (Moro et al., 2014)

#### Modelos comparados y resultado final

| Modelo | AUC (modeling) | ALIFT (modeling) | AUC (real evaluation) | ALIFT (real) |
|---|---|---|---|---|
| **Logistic Regression** | 0.900 | 0.849 | 0.715 |0.626 |
| **Decision Trees** | 0.833 | 0.756 | 0.757 | 0.651 |
| **SVM** | 0.891 | 0.844 | 0.767 | 0.656 |
| **Neural Network** | 0.929 | 0.878 | 0.794 | 0.672 |

Las Redes Neuronales parecen ser superiores en predicción y generalización (con rolling window y sin usar datos del futuro).

#### Métricas de evaluación recomendadas
1. **AUC (Area Under the ROC Curve):** mide discriminación global del modelo.
2. **ALIFT:** mide ganancia respecto a targeting aleatorio (muy usada en marketing).
- También se recomienda mostrar curvas ROC y Lift, y confusion matrix para cierto umbral.
- Se menciona que el dataset original tiene fuerte desbalance (≈12% positivos)

*“AUC is a popular classification metric that presents advantages of being independent of the class frequency or specific false positive/negative costs.”* — Sección 2.3

*“In the domain of marketing, the Lift analysis is popular for accessing the quality of targeting models.”* — Sección 2.3

Las métricas AUC y ALIFT  permiten evaluar la capacidad predictiva de los modelos sin estar influenciadas por el desbalance de clases (AUC), y porque ALIFT es especialmente útil en campañas de marketing al mostrar el valor real de priorizar clientes con mayor probabilidad de conversión. Indica cuántas veces más clientes positivos se pueden encontrar en los percentiles más altos comparado con targeting aleatorio.

*”…79% of the successful sells can be achieved when contacting only half of the clients…”* — Sección 3.2

Antes de leer el paper, me hubiera decantado por las métricas clásicas, como la `precision`, el `recall` y el `F1 score`. Sin embargo, en este contexto de marketing y sobre todo con un dataset desbalanceado (solo ≈12% de positivos), estas métricas pueden no ser las más adecuadas para la optimización del negocio.
Si bien son útiles para tareas donde los errores tienen costos fijos (como lo puede ser un diagnóstico médico), su utilidad se ve limitada en este escenario en el que no se toma una decisión binaria (contactar o no), el objetivo es priorizar a los clientes más propensos a convertir.

Esto es fundamental, ya que en marketing el objetivo no es necesariamente clasificar perfectamente, sino **ordenar clientes según su probabilidad de conversión** y actuar sobre los mejores candidatos. En este sentido, métricas como F1 podrían incluso degradar la calidad del ranking que se busca.

Por tanto, maximizar AUC y ALIFT ofrece un enfoque más realista y alineado con los objetivos del negocio.


#### Variables relevantes
La fuente menciona algunas variables no proporcionadas en esta versión del dataset: `call.dir`, `ag.created`, `dif.best.rate.avg`

## 3. EDA

#### Relación con la variable y

| Variable | Observación clave |
|---|---|
| ` job ` | `student` y `retired` destacan con tasas altas de suscripción (31%, 25%) |
| ` marital ` | `single` tiene mayor tasa de yes (14%) que `married` o `divorced` (~10%) |
| ` education ` | `illiterate` y `university.degree` con tasas elevadas (22%, 13.7%) |
| ` default ` | `yes` no tiene ningún suscriptor y < 1% de ocurrencia |
| ` contact ` | `cellular` mucho más efectivo que telephone |
| ` month ` | `mar`, `dec`, `sep`, `oct` tienen tasas muy altas (>40%) |
| ` poutcome ` | Si fue `success` antes, gran probabilidad de éxito ahora (65%) |

#### Decisiones clave
- Eliminación de 990 registros donde `housing` y `loan` eran simultáneamente ‘unknown’, ya que representaban un patrón redundante sin valor predictivo claro. Su proporción de clase y es similar a la del dataset global, por lo que no afecta la distribución del target.
- Eliminación de 79 registros con `marital` == 'unknown' (0.2%), al no representar una clase significativa ni ofrecer valor predictivo diferencial respecto al resto de categorías. Además, el estado civil es realmente relevante desde un punto de vista social.
- Imputación sobre `education` en función de `employment_status` (variable generada con mapeo sobre `job`), dado que el 71% de los casos con `education` == "unknown" pertenecían al grupo employed. También usé la moda de `education` dentro de cada grupo de `employment_status`.
- Imputación de los valores faltantes en la variable `job` (316 registros, ≈ 0.79% del total) se realizó con base en dos variables relevantes: `education` y ¬, debido a su fuerte asociación con `job` según la matriz de Cramér’s V: `education = 0.36`, `marital = 0.22`
- Dado su comportamiento, `pdays` no es una variable numérica continua clásica, sino una pseudo-categórica con dos grupos claros que además demostraban distribución diferente respecto a la variable target. Se convirtió a categórica. Algo similar hice con had_previous_contact.

## 4. Elección de Features
La selección final de variables se fundamentó en dos enfoques complementarios:

1. **Dependencia estadística**: usé matriz de Cramér's V para evaluar la asociación entre variables categóricas y la variable objetivo (`y`). Se seleccionaron aquellas con mayor dependencia como `contact`, `poutcome`, y múltiples categorías dentro de `job` y `month`.

2. **Correlación lineal**: Para variables numéricas, analicé la correlación de Pearson respecto a `y`. Las más destacadas fue `nr.employed` con correlación significativas y consistente con el comportamiento observado durante el análisis exploratorio. `cons.conf.idx` no pareció tener mucha correlación pero al agregarla el modelo aumentó ligeramente su desempeño, al no tener datos nulos, decidí que valía la pena agregarla a la solución.

Además:
- Las variables categóricas fueron transformadas mediante one-hot encoding o label encoding dependiendo de lo observado, manteniendo categorías relevantes basadas en su aporte predictivo.
- `pdays` fue transformada en dos variables categóricas: `pdays_was_contacted` y `contact_type`, reflejando si el cliente había sido contactado previamente y si fue contacto reciente o lejano. la decisión fue `contact_type` para evitar problemas de multicolinealidad con las otras 2.

Más adelante el uso de SHAP permitirá validar la importancia de las variables en el mejor modelo (si es que aplica).

## 5. Construcción de Modelos
Construcción y comparación de 3 algoritmos diferentes: Regresión Logística (Logistic Regression), Random Forest (RF) y HistGradientBoosting (HistGB). Se aplicó validación cruzada anidada (Nested CV) con búsqueda de hiperparámetros mediante GridSearchCV. La métrica de optimización fue AUC y se reportaron tanto AUC como ALIFT para evaluación.

Performance Summary (Mean ± Std):

| Modelo  | AUC Mean | AUC Std | ALIFT Mean | ALIFT Std |
|---------|----------|---------|------------|-----------|
| Logistic Regression  | 0.7900   | 0.0109  | 2.1993     | 0.0492    |
| RF      | 0.7907   | 0.0107  | 2.2010     | 0.0451    |
| HistGB  | 0.7934   | 0.0116  | 2.2331     | 0.0594    |

Most Common Hyperparameters:

- Logistic Regression: `{'C': 10}` (10 times)
- RF: `{'max_depth': 5, 'n_estimators': 100}` (8 times)
- HistGB: `{'learning_rate': 0.01, 'max_leaf_nodes': 15}` (7 times)

#### Selección de Modelos

- **Regresión Logística**: Es un modelo base interpretable y eficiente en datasets con muchas variables categóricas. Su desempeño suele ser competitivo en problemas lineales. El parámetro `C` controla la regularización y en el GridSearch explora valores de baja a moderada penalización: `[0.01, 0.1, 1, 10]`. `class_weight='balanced'` para manejar el fuerte desbalance del target (≈12% positivos).

- **Random Forest (RF)**: Ensamble de árboles, robusto ante ruido y capaz de capturar relaciones no lineales y efectos de interacción entre variables. Igual`class_weight='balanced'` para mejorar su sensibilidad sobre la clase minoritaria. Se ajustan `n_estimators` (cantidad de árboles) y `max_depth` (profundidad máxima) para balancear capacidad de aprendizaje y sobreajuste. La combinación elegida permite modelar estructuras complejas sin perder generalización.

- **HistGradientBoosting (HistGB)**: Versión eficiente del Gradient Boosting para datasets moderadamente grandes. Es menos sensible al preprocesamiento y funciona bien con variables categóricas codificadas. Se ajusta `learning_rate` (que controla la velocidad de aprendizaje) y `max_leaf_nodes` (complejidad del árbol base).

Esta combinación la escogí para tener un modelo lineal, uno basado en bagging y uno en boosting, cubriendo así un espectro amplio de estrategias predictivas.


#### Pipeline de Evaluación

El pipeline de evaluación está diseñado para comparar modelos de forma robusta y justa, alineándose con las métricas clave previamente definidas: **AUC** y **ALIFT**, ambas independientes del desbalance de clases presente en este caso y altamente interpretables en contextos de marketing.

1. **Separación de features y target**:  
   `X = df_encoded[features]` y `y = df_encoded[target]`. Posteriormente, las variables predictoras se escalan usando `StandardScaler` para asegurar que los modelos sensibles a la escala (como la regresión logística) funcionen de manera adecuada.

2. **Partición del conjunto de datos**:  
   Se aplica un `train_test_split` estratificado (75% train, 25% test) para conservar la proporción de clases. Esta división permite hacer una validación externa posterior con el mejor modelo entrenado.

3. **Definición de modelos y búsqueda de hiperparámetros**:  
   Se comparan los tres algoritmos, cada uno con una cuadrícula de hiperparámetros relevante y `class_weight='balanced'` donde aplica, para lidiar con el desbalance.

4. **Validación cruzada anidada (Nested CV)**:  
    Ejecución una validación cruzada externa de 10 (en este caso, la decisión fue empírica) folds (StratifiedKFold) y dentro de cada fold se aplica `GridSearchCV` para encontrar los mejores hiperparámetros optimizando AUC. Este enfoque reduce el riesgo de overfitting al hiperparámetro y permite obtener métricas robustas.

5. **Evaluación de métricas**:
   - **AUC** se calcula directamente con `roc_auc_score`.
   - **ALIFT** se calcula con una función que integra el área bajo la curva de ganancia acumulativa (Lift).
   - **boxplots** para visualizar la distribución de resultados y **heatmaps con el test de Nemenyi** para comparar estadísticamente los modelos.

6. **Selección del mejor modelo**:  
   `HistGradientBoostingClassifier` mostró desempeño superior en AUC y ALIFT, evaluado tanto por la media como por el ranking.

7. **Evaluación final y explicabilidad**:  
   Reentreno el mejor modelo en el conjunto de entrenamiento y se evalúa en el conjunto de prueba (hold-out) generando:
   - Matriz de confusión
   - Curva ROC
   - Cumulative Lift Curve y ALIFT
   - Interpretabilidad vía **SHAP** para entender las contribuciones de cada variable.


## 6. Evaluación de Modelos
El modelo final, después de la exploración, procesamiento, selección de features y validación cruzada fue:
```python
final_model = HistGradientBoostingClassifier(max_iter=300,
            early_stopping=False,
            learning_rate=0.01,
            max_leaf_nodes=15,
            random_state=42)
```

Como ya fue mencionado, el mejor modelo fue HistGradientBoostingClassifier, con un desempeño consistente y superior.

- **Consistencia estadística**: Más allá de la media, los boxplots reflejaron una distribución favorable en comparación con los otros modelos, y el test de Nemenyi confirmó que las diferencias fueron estadísticamente significativas:
  - En términos de **AUC** (Score: 0.8058), HistGB superó significativamente la Logistic Regression y Random Forest, respaldado también con una prueba de hipótesis p 0.05.
  - En **ALIFT**, la mejora fue clara frente a ambos modelos: p ≈ 0.005 (vs Logistic Regression) y p ≈ 0.01 (vs Random Forest).
- **Curva ROC y AUC:** destaca su habilidad discriminativa (AUC ≈ 0.8058).
- **Lift y Cumulative Lift Curve:**  indican que el modelo identifica eficazmente a los clientes con alta probabilidad de conversión. El ALIFT alcanzado fue de 0.7714, lo cual significa que el modelo es un 77.14% más efectivo que un targeting aleatorio al momento de captar clientes potenciales. Esto implica que, al usar el modelo para priorizar a quién contactar, se puede obtener una ganancia significativa en eficiencia de campañas.
Además, el lift en el top 10% fue de 4.63, lo que indica que si se contacta al 10% de clientes con mayor score del modelo, se obtienen 4.631 veces más suscriptores que si se hubieran contactado al azar. (2.28 sobre el total del dataset, que sigue siendo muy buen desempeño)
- **Interpretabilidad**: Permite análisis de importancia de variables con SHAP, lo cual es esencial en escenarios de negocio como marketing financiero, donde se requiere entender las decisiones del modelo.
- **Robustez y escalabilidad**: HistGB es eficiente para datasets de tamaño moderado, tolerante a outliers y capaz de modelar relaciones no lineales sin necesidad de preprocesamiento complejo. A diferencia de modelos tradicionales de boosting, su versión basada en histogramas permite reducir tiempos de entrenamiento sin sacrificar rendimiento.

#### Interpretación SHAP
Es una técnica basada en teoría de juegos que permite cuantificar la contribución de cada variable a las predicciones individuales del modelo.
En este caso, el gráfico SHAP resume el impacto de las 20 variables más importantes del modelo HistGradientBoostingClassifier, mostrando para cada observación (un punto en el gráfico):
1. El color representa el valor de la característica: rojo indica un valor alto y azul un valor bajo dentro del rango de la variable.
2.  La posición en el eje X (SHAP value) indica cuánto ese valor empujó la predicción hacia el sí (derecha) o hacia el no (izquierda).

**Hallazgos clave:**
- `nr.employed`: Valores altos de esta variable (mayor número de empleados promedio en el mercado) tienen un efecto negativo en la probabilidad de suscripción. Esto podría deberse a que campañas en contextos de mayor estabilidad laboral pueden generar menor interés en productos financieros adicionales.
- `contact`: Tener contacto vía celular impacta positivamente en la predicción. Esto es consistente con el análisis exploratorio, donde las tasas de conversión eran mucho mayores para llamadas celulares que telefónicas tradicionales.
- `cons.conf.idx` (Índice de confianza del consumidor): Aunque inicialmente parecía tener baja correlación con y, el modelo aprendió que valores bajos del índice (que indican pesimismo económico) están asociados a la probabilidad de suscripción.
- `pdays_was_contacted`: Contactos previos recientes en una campaña tienen un fuerte efecto positivo en la predicción, lo que valida su inclusión como variables derivadas.
- `poutcome_success` y `poutcome_failure`: El resultado de campañas anteriores sigue siendo un fuerte predictor. Como se observa, haber tenido éxito previamente tiene un gran impacto positivo, mientras que el fracaso tiende a empujar la predicción hacia un no.
- `month_oct`: curiosamente octubre es el mes que más resalta, probablemente por estacionalidad o características propias de las campañas lanzadas en esos periodos.

## 7. Conclusiones
Respondiendo a las preguntas de las instrucciones:
1. ¿Qué clientes tienen mayor probabilidad de contratar?
- 	Aquellos que fueron contactados en campañas previas exitosas (`poutcome_success`), lo que sugiere interés y familiaridad con el producto.
- Recibieron contacto a través de celular (`contact = cellular`), ser contactados por teléfono fijo suele ser incluso contraproducente y pesa mucho en el no.
- Hubo contacto reciente (`pdays_was_contacted = True`), lo cual muestra que la persistencia puede tener efecto positivo.
- Están en meses con estacionalidad favorable, sobre todo octubre. Históricamente saca tasas más altas de conversión.
- Tienen menor número de empleados promedio en el mercado laboral (`nr.employed` bajo), lo cual puede reflejar contextos económicos donde el cliente busca proteger o maximizar su dinero.

2. ¿Cómo puede usar esto el área de marketing?
- **Segmentación proactiva:** En lugar de contactar a toda la base de clientes, el modelo permite priorizar a aquellos con mayor probabilidad de conversión. Por ejemplo, contactar solo al 10% superior según el score del modelo multiplica por 4.5 la tasa de conversión respecto a hacerlo de forma aleatoria.
- **Optimización de campañas:** Permite planear campañas en meses más efectivos, usando medios más exitosos como el celular y focalizándose en clientes con historial positivo.
- **Personalización del mensaje:** Incluso desde el EDA se pueden notar ciertos patrones, el equipo puede adaptar los mensajes según las características del cliente. Por ejemplo, reforzar beneficios específicos a jubilados o estudiantes, que son aquellos con ya de por si una tasa alta de suscripción; o tratar de atacar ocupaciones que tienen la tasa baja para que esta aumente. 
- **Evaluación continua:** Al integrar nuevas campañas, el modelo puede seguir actualizándose y ayudar a detectar cambios en comportamiento o respuesta, manteniéndose como una herramienta viva y útil.

Nota Final: Modelo adjunto en `model/final_model.joblib`

## 8. Fuentes
S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems (2014), doi:10.1016/j.dss.2014.03.001.