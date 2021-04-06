# Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK

![Logo](https://static.platzi.com/media/achievements/badge-fundamentos-procesamiento-lenguaje-natural-python-85faceb3-d6e5-4829-9c80-847.png)

## Profesor

Francisco Camacho

## Archivos

* [Slides](/files/slides.pdf)
* [Notebook: Configuración Quickstart NLTK](/code/01_configuracion_quickstart.ipynb)
* [Notebook: Texto y vocabulario con estadística](/code/02_text_vocab_stats.ipynb)
* [Notebook: Lenguaje con estadística](/code/03_lenguaje_con_estadistica.ipynb)
* [Notebook: Recursos Léxicos](/code/04_recursos_lexicos.ipynb)

## Índice

1. [Introducción al Procesamiento de Lenguaje Natural](#introducción-al-procesamiento-de-lenguaje-natural)
    * [Introducción al Procesamiento de Lenguaje Natural](#introducción-al-procesamiento-de-lenguaje-natural-1)
    * [Evolución del NLP](#evolución-del-nlp)
    * [Conceptos básicos de NLP](#conceptos-básicos-de-nlp)
2. [Fundamentos con NLTK](#fundamentos-con-nltk)
    * [Configurar ambiente de trabajo](#configurar-ambiente-de-trabajo)
    * [Palabras, textos y vocabularios](#palabras-textos-y-vocabularios)
    * [Tokenizacion con Expresiones Regulares](#tokenizacion-con-expresiones-regulares)
    * [Estadísticas básicas del lenguaje](#estadísticas-básicas-del-lenguaje)
    * [Distribuciónes de frecuencia de palabras](#distribuciónes-de-frecuencia-de-palabras)
    * [Refinamiento y visualización de cuerpos de texto](#refinamiento-y-visualización-de-cuerpos-de-texto)
    * [N-gramas y Colocaciones del lenguaje](#n-gramas-y-colocaciones-del-lenguaje)
    * [¿Cómo extraer n-gramas de un texto en Python?](#cómo-extraer-n-gramas-de-un-texto-en-python)
    * [Colocaciones en Python](#colocaciones-en-python)
    * [Colocaciones en gráficos de dispersión](#colocaciones-en-gráficos-de-dispersión)
    * [Filtros y colocaciones en NLTK](#filtros-y-colocaciones-en-nltk)
    * [Introducción a los recursos léxicos](#introducción-a-los-recursos-léxicos)
    * [Recursos léxicos en NLTK](#recursos-léxicos-en-nltk)
    * [NLTK para traducción de palabras](#nltk-para-traducción-de-palabras)

---

# Introducción al Procesamiento de Lenguaje Natural

## Introducción al Procesamiento de Lenguaje Natural

### ¿Por qué es tan importante el procesamiento de lenguaje natural?

Por que creemos que es el autentico camino de lo que creemos que es el ideal de lo que nosotros consideramos inteligencia artificial?

### ¿Qué significa y de que se encarga el NLP?

Significa Natural Language Processing y este es un área que combina ciencias de la computación, lingüística, IA para entender como se pueden ejecutar interacciones entre humanos y maquinas por medio le lenguaje natural.

### ¿Qué significa y de que se encarga el NLU?

Significa Natural Language Understanding y esta es una sub área del NLP se encarga de tareas especificas que las maquinas puedan ejecutar en el proceso de comunicación de los seres humanos de manera que esas tarea reflejen que el robot no solo puede procesar nuestro lenguaje, si no entenderlo y las respuestas que nos dé, deben de reflejar que verdaderamente lo entiende.

### ¿Qué es el test de Turing?

Es un examen de la capacidad de una máquina para exhibir un comportamiento inteligente similar al de un ser humano o indistinguible de este. Alan Turing propuso que un humano evaluara conversaciones en lenguaje natural entre un humano y una máquina diseñada para generar respuestas similares a las de un humano. El evaluador sabría que uno de los participantes de la conversación es una máquina y los intervinientes serían separados unos de otros. La conversación estaría limitada a un medio únicamente textual como un teclado de computadora y un monitor por lo que sería irrelevante la capacidad de la máquina de transformar texto en habla. En el caso de que el evaluador no pueda distinguir entre el humano y la máquina acertadamente (Turing originalmente sugirió que la máquina debía convencer a un evaluador, después de 5 minutos de conversación, el 70 % del tiempo), la máquina habría pasado la prueba. Esta prueba no evalúa el conocimiento de la máquina en cuanto a su capacidad de responder preguntas correctamente, solo se toma en cuenta la capacidad de esta de generar respuestas similares a las que daría un humano.

> ... Si un humano no puede distinguir entre una máquina y otra persona en una conversación, entonces esa máquina ha alcanzado un nivel de inteligencia comparable al de un humano... Alan Turing.

**Usos actuales del NLP:**

* Máquinas de búsqueda o motores de búsqueda.
* Traductores de texto.
* Chatbots.
* Análisis de discurso.
* Reconociendo del habla.

**¿Por qué es tan difícil el NLP?**

* El problema mas grande son las ambigüedades.
* Es difuso.
* Requiere de mucho contexto.

## Evolución del NLP

![Evolución NLP](/images/evolucion-nlp.PNG)

![Avances NLP](/images/avances-nlp.PNG)

![Avances NLP 2](/images/avances-nlp1.PNG)

> El trofeo no cabe en la caja porque es muy grande. El trofeo no cabe en la caja porque es muy pequeña.

En el ejemplo anterior podemos ver que la primera oración hace referencia a que el trofeo es muy grande y por ello no cabe en la caja, mientras que en la segunda oración hace referencia a que la caja es muy pequeña y por eso el trofeo cabe en ella. Ambas dicen exactamente lo mismo, pero los algoritmos de NLP no comprendieron lo que significaba.

### Qué vamos a estudiar

![Qué estudiar](/images/que-vamos-a-estudiar.PNG)

Vamos a utilizar las librerías nltk y spacy y exploraremos los 3 grandes bloques de NLP.

[SOBRE LA CONFIGURACIÓN ESTADÍSTICA DE LOS CORPUS TEXTUALES](http://www.scielo.edu.uy/scielo.php?script=sci_arttext&pid=S2079-312X2017000100121)

## Conceptos básicos de NLP

> Entender y caracterizar las reglas que determinan cómo estructurar expresiones lingüísticas … Manning & Schütze (1999), Foundations of Statistical Natural Language Processing

**NLP**: El procesamiento de lenguaje natural esta más enfocado hacia aplicaciones practicas en la ingeniería

**LC**: La lingüística computacional estudia el lenguaje desde una perspectiva más científica. (Basada en crear modelos que pueden tener dos enfoques de conocimiento o datos)

El procesamiento de una cadena de texto necesita una Normalización que incluye los siguientes procesos:

* Tokenización: Separar en palabras toda la cadena de texto. *Mi hermano dejó de comer -> |Mi|hermano|dejó|de|comer|*
* Lematización: Convertir cada una de las palabras a su raiz fundamental. *Mi hermano **dejó** de comer -> Mi hermano **dejar** de comer.*
* Segmentación: Separación en frases (puede ser con las comas). *Mi hermano dejó de comer, no se sentía muy bien -> |Mi hermano dejó de comer|no se sentía muy bien|*.

* **CORPUS**: Colección de muchos textos
* **CORPORA**: Colección de colecciones de texto

[Foundations of Statistical Natural Language Processing](/files/Manning_Schuetze_StatisticalNLP.pdf)

# Fundamentos con NLTK

## Configurar ambiente de trabajo

### Corpus lingüístico
Un corpus lingüístico es un conjunto amplio y estructurado de ejemplos reales de uso de la lengua. Estos ejemplos pueden ser textos, o muestras orales.​ Un corpus lingüístico es un conjunto de textos relativamente grande, creado independientemente de sus posibles formas o usos.

### Token
Un token es un conjunto de caracteres que representan texto. También podemos decir que el token es la unidad análisis de texto, así como un número es la unidad del análisis matemático. Es fácil para nosotros pensar que un token es igual a una palabra, sin embargo esto no es correcto, puesto que la “palabra” es un elemento del lenguaje que posee significado por sí misma, mientras que el token se supone es un elemento abstracto. Dependiendo de la tarea que estemos afrontando, el token puede ser alguna de las siguientes:

* Una sola palabra, como: “jóvenes”, “nivel” o “superior”,
* Un número, como: “1”, “0”, o “10”,
* Un solo caracter, como: “j”, “ó” o “v”,
* Un símbolo, como “¿”, “?” o “#”,
* Un conjunto de caracteres, como “nivel superior” o “escuela técnica”

### Tokenización
La tokenización es un paso que divide cadenas de texto más largas en piezas más pequeñas o tokens. Los trozos de texto más grandes pueden ser convertidos en oraciones, las oraciones pueden ser tokenizadas en palabras, etc. El procesamiento adicional generalmente se realiza después de que una pieza de texto ha sido apropiadamente concatenada. La tokenización también se conoce como segmentación de texto o análisis léxico. A veces la segmentación se usa para referirse al desglose de un gran trozo de texto en partes más grandes que las palabras (por ejemplo, párrafos u oraciones), mientras que la tokenización se reserva para el proceso de desglose que se produce exclusivamente en palabras.

## Palabras, textos y vocabularios

* **Vocabulario**: Son las palabras únicas en un corpus.

![Cheat sheet REGEX](/images/regex.jpg)

## Tokenizacion con Expresiones Regulares

* Imprimimos el texto con un salto de linea
```py
print('esta es \n una prueba')
```
* Forzamos a Python a que nos imprima el texto como texto plano a pesar de los caracteres especiales como el salto de linea. 
```py
print(r'esta es \n una prueba')
```
* **Tokenización**: Es el proceso mediante el cual se sub-divide una cadena de texto en unidades lingüísticas mínimas (palabras)
```py
# Ejemplo: 

texto = """ Cuando sea el rey del mundo (imaginaba él en su cabeza) no tendré que  preocuparme por estas bobadas. 
            Era solo un niño de 7 años, pero pensaba que podría ser cualquier cosa que su imaginación le permitiera 
            visualizar en su cabeza ...""" 
# Caso 1: Tokenizar por espacios vacíos
print(re.split(r' ',texto))
# output: ['', 'Cuando', 'sea', 'el', 'rey', 'del', 'mundo', '', '(imaginaba', 'él', 'en', 'su', 'cabeza)', 'no', 'tendré', 'que', '', 'preocuparme', 'por', 'estas', 'bobadas.', '\n', '', '', '', '', '', '', '', '', '', '', '', 'Era', 'solo', 'un', 'niño', 'de', '7', 'años,', 'pero', 'pensaba', 'que', 'podría', 'ser', 'cualquier', 'cosa', 'que', 'su', 'imaginación', 'le', 'permitiera', 'visualizar', 'en', 'su', 'cabeza', '...']

# Caso 2: Tokenizar usado expresiones regulares usando regex
print(re.split(r'[ \t\n]+', texto))
# output: ['', 'Cuando', 'sea', 'el', 'rey', 'del', 'mundo', '(imaginaba', 'él', 'en', 'su', 'cabeza)', 'no', 'tendré', 'que', 'preocuparme', 'por', 'estas', 'bobadas.', 'Era', 'solo', 'un', 'niño', 'de', '7', 'años,', 'pero', 'pensaba', 'que', 'podría', 'ser', 'cualquier', 'cosa', 'que', 'su', 'imaginación', 'le', 'permitiera', 'visualizar', 'en', 'su', 'cabeza', '...']

# Caso 3: Usando el parámetro W
print(re.split(r'[ \W\t\n]+', texto))
# output: ['', 'Cuando', 'sea', 'el', 'rey', 'del', 'mundo', 'imaginaba', 'él', 'en', 'su', 'cabeza', 'no', 'tendré', 'que', 'preocuparme', 'por', 'estas', 'bobadas', 'Era', 'solo', 'un', 'niño', 'de', '7', 'años', 'pero', 'pensaba', 'que', 'podría', 'ser', 'cualquier', 'cosa', 'que', 'su', 'imaginación', 'le', 'permitiera', 'visualizar', 'en', 'su', 'cabeza', '']

```
### Tokenizando con NLTK
```py
texto = 'En los E.U. esa postal vale $15.50 ...'
print(re.split(r'[ \W\t\n]+', texto))

pattern = r'''(?x)                  # Flag para iniciar el modo verbose
              (?:[A-Z]\.)+            # Hace match con abreviaciones como U.S.A.
              | \w+(?:-\w+)*         # Hace match con palabras que pueden tener un guión interno
              | \$?\d+(?:\.\d+)?%?  # Hace match con dinero o porcentajes como $15.5 o 100%
              | \.\.\.              # Hace match con puntos suspensivos
              | [][.,;"'?():-_`]    # Hace match con signos de puntuación
              '''
nltk.regexp_tokenize(texto, pattern)
# Output: ['en', 'los', 'E.U.', 'esa', 'postal', 'vale', '$15.50', '...']
```

[Archivo Google Colab](https://colab.research.google.com/drive/1rFKNA6wp5E6SIvJTuNy1dulhrL8Gbdkt?usp=sharing)

[Archivo local](/code/01_configuracion_quickstart.ipynb)

## Estadísticas básicas del lenguaje

### Riqueza léxica

La riqueza léxica se define como la relación entre el número de palabras únicas en un texto y el total de palabras del texto.

![Riqueza léxica](/images/riqueza-lexica.PNG)

Otra medida que suele ser interesante, es el porcentaje de veces que se repite una palabra en un texto.

```py
def riqueza_lexica(texto):
    vocabulario = sorted(set(texto))
    return len(vocabulario)/len(texto)

import re
texts = [ i for i in dir(nltk.book) if re.search(r'text\d', i)]
for text in texts:
    exec(compile(f'print({text}.name, "\\n", riqueza_lexica({text}), "\\n" )', 
        '', 'exec'))
"""
Moby Dick by Herman Melville 1851 
 0.07406285585022564 

Sense and Sensibility by Jane Austen 1811 
 0.04826383002768831 

The Book of Genesis 
 0.06230453042623537 

Inaugural Address Corpus 
 0.06617622515804722 

Chat Corpus 
 0.13477005109975562 

Monty Python and the Holy Grail 
 0.1276595744680851 

Wall Street Journal 
 0.12324685128531129 

Personals Corpus 
 0.22765564002465585 

The Man Who Was Thursday by G . K . Chesterton 1908 
 0.0983485761345412 
"""
```

## Distribuciónes de frecuencia de palabras

Los cálculos estadísticos más simples que se pueden efectuar sobre un texto o un corpus son los relacionados con frecuencia de aparición de palabras.

* Podemos construir un diccionario en Python donde las llaves sean las palabras y los valores sean las frecuencias de ocurrencias de esas palabras.

* ejemplo `dic = {'monster': 49 ,  'boat': 54,  ...}`

Una forma básica de construir el diccionario es la siguiente:
```py
dic = {}

for palabra in set(text1):
    dic[palabra] = text1.count(palabra)
dic[:10]
```
Sin embargo, esto no es recomendable, porque son más muchas palabras únicas y tiene que contar cuantas veces se repite cada una de estas.

NLTK tiene una función que se encarga de hacer el conteo de una forma más eficiente, se llama FreqDist.
```py
fdist = FreqDist(text1)
fdist.most_common(20)
"""
[(',', 18713),
 ('the', 13721),
 ('.', 6862),
 ('of', 6536),
 ('and', 6024),
 ('a', 4569),
 ('to', 4542),
 (';', 4072),
 ('in', 3916),
 ('that', 2982),
 ("'", 2684),
 ('-', 2552),
 ('his', 2459),
 ('it', 2209),
 ('I', 2124),
 ('s', 1739),
 ('is', 1695),
 ('he', 1661),
 ('with', 1659),
 ('was', 1632)]
"""
fdist['monster']
# 49
fdist.plot(30)
```
![Plot](/images/fdist.png)

En realidad, analizando este tipo de datos se encuentran aspectos relacionados con los sistemas complejos. Un ejemplo es que tanto en el libro de Moby Dick como en cualquier obra ( un periódico, la biblia, etc.) la distribución de palabras sigue una ley de potencias (o power law). Una explicación a este fenómeno se da con la criticalidad auto-organizada. Para verificar si se cumple esta ley de potencias se debe verifica una tendencia linear graficando en escala logarítmica ambos ejes (un gráfico log-log).

Esto se conoce como la ley de Zipf. En el libro “How nature works” de Per Bak muestra este ejemplo:

![Log](/images/log-log.webp)

Una forma de tener el gráfico log-log de la distribución de palabras en el libro “Moby Dick” es (cuando ya tenemos el fdist) usando:
```py
a=list(fdist.values())
a=np.array(a)
a=np.sort(a)
a=a[::-1]
plt.yscale('log')
plt.xscale('log')
plt.plot(a)
```
![Moby dick](/images/log-log-moby-dick.png)

## Refinamiento y visualización de cuerpos de texto

### Distribuciones sobre contenido con filtro-fino

* Como vimos en la sección anterior, los tokens más frecuentes en un texto no son necesariamente las palabras que mas informacion nos arrojan sobre el contenido del mismo. 
* Por ello, es mejor filtrar y construir distribuciones de frecuencia que no consideren signos de puntuación o caracteres especiales


[Archivo Google Colab](https://colab.research.google.com/drive/1_35vfOLiNI-0Gulg0EpOM2lXfI6OIHMO?usp=sharing)

[Archivo local](/code/02_text_vocab_stats.ipynb)

## N-gramas y Colocaciones del lenguaje

Es una secuencia de n palabras consecutivas.

### Bi-gramas

Secuencia de 2 palabras consecutivas.

*Estoy aprendiendo cosas increíbles* -> `(Estoy, aprendiendo), (aprendiendo, cosas), (cosas, increíbles)`.

### Tri-gramas

Secuencia de 3 palabras consecutivas.

*Estoy aprendiendo cosas increíbles* -> `(Estoy, aprendiendo, cosas), (aprendiendo, cosas, increíbles)`.

### Colocaciones

> Las colocaciones de una palabra son sentencias que indican los lugares que acostumbra a tomar esa palabra en el lenguaje (sin seguir las reglas del lenguaje) .... Firth (1957), Modes in Meaning - Paper in Linguistics

*Le dieron ganas de dormir. Le introdujeron ganas de dormir*. Estas 2 oraciones dicen exactamente lo mismo, sin embargo, la segunda suena menos natural en el lenguaje. La primera suena más natural, PERO, no hay una regla específica o concreta en el lenguaje que nos diga que es correcto usar 'dieron' en vez de 'introdujeron'. Técnicamente ambas están igual de usadas de usadas en un sentido equivocado, pero por razones culturales usamos la palabra dieron, esto es una manera de colocación, Es una palabra que aparece con una frecuencia inusual en una cierta ubicación de las frases y esto no tiene una explicación basada en una regla del lenguaje.

*Ventilar secretos*. Es una frase común en países hispano-hablantes, quiere decir que estás revelando un secreto que se supone no se debe decir a nadie.

Debido a que las colocaciones están basadas en la cultura no existen reglas estrictas en el lenguaje para que se utilicen de la manera que se hacen, sin embargo, vamos a ver estadísticas que nos permiten identificar de forma numérica usando colocaciones del lenguaje.

## ¿Cómo extraer n-gramas de un texto en Python?

Primero importamos las librerías que vamos a utilizar.
```py
import nltk
nltk.download('book')
from nltk.book import * 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
```

**Bi-gramas**: Parejas de palabras que ocurren consecutivas.

Creamos los bigramas y utilizamos FreqDist para hallar la distribución de frecuencias.
```py
md_bigrams = list(bigrams(text1))
fdist = FreqDist(md_bigrams)
fdist.most_common(20)
"""
[((',', 'and'), 2607),
 (('of', 'the'), 1847),
 (("'", 's'), 1737),
 (('in', 'the'), 1120),
 ((',', 'the'), 908),
 ((';', 'and'), 853),
 (('to', 'the'), 712),
 (('.', 'But'), 596),
 ((',', 'that'), 584),
 (('.', '"'), 557),
 ((',', 'as'), 523),
 ((',', 'I'), 461),
 ((',', 'he'), 446),
 (('from', 'the'), 428),
 ((',', 'in'), 402),
 (('of', 'his'), 371),
 (('the', 'whale'), 369),
 (('.', 'The'), 369),
 (('and', 'the'), 357),
 ((';', 'but'), 340)]
 """
 ```
 Graficamos la distribución actual.
```py
fdist.plot(20)
```
![FreqDist most common](/images/freqdist-common.png)

Sin embargo esto no es muy eficiente, puesto que cuenta con conectores y signos de puntuación, por lo tanto, debemos filtrarlos y aplicar nuevamente la distribución de frecuencias.

### Filtrado de bi-gramas

* Sin embargo, observamos que los bi-gramas más comunes no representan realmente frases o estructuras léxicas de interés.
* Tal vez, aplicar algún tipo de filtro nos permita ver estructuras relevantes
```py
threshold = 2
filtered_bigrams = [bigram for bigram in md_bigrams if len(bigram[0])>threshold and len(bigram[1])>threshold]
filtered_dist = FreqDist(filtered_bigrams)
filtered_dist.most_common(20)
"""
[(('from', 'the'), 428),
 (('the', 'whale'), 369),
 (('and', 'the'), 357),
 (('with', 'the'), 308),
 (('for', 'the'), 285),
 (('into', 'the'), 246),
 (('the', 'ship'), 235),
 (('the', 'sea'), 223),
 (('upon', 'the'), 216),
 (('that', 'the'), 215),
 (('all', 'the'), 198),
 (('the', 'same'), 159),
 (('the', 'Pequod'), 147),
 (('the', 'other'), 135),
 (('over', 'the'), 133),
 (('and', 'then'), 129),
 (('have', 'been'), 122),
 (('Sperm', 'Whale'), 118),
 (('the', 'boat'), 118),
 (('had', 'been'), 115)]
"""
```
Graficamos con `filtered_dist.plot(20)`.

![Filtered dist](/images/filtered-freqdist.png)

### Tri-gramas
```py
from nltk.util import ngrams
md_trigrams = list(ngrams(text1, 3))
fdist = FreqDist(md_trigrams)
print(fdist.most_common(20))
fdist.plot(20)
"""
[((',', 'and', 'the'), 187), (('don', "'", 't'), 103), (('of', 'the', 'whale'), 101), ((',', 'in', 'the'), 93), ((',', 'then', ','), 87), (('whale', "'", 's'), 81), (('.', 'It', 'was'), 81), (('ship', "'", 's'), 80), (('the', 'Sperm', 'Whale'), 77), ((',', 'as', 'if'), 76), (('he', "'", 's'), 76), (('Ahab', "'", 's'), 75), (('.', 'Now', ','), 74), (("'", 's', 'a'), 73), (("'", 's', 'the'), 72), (('that', "'", 's'), 69), ((',', 'as', 'the'), 68), (('the', 'sea', ','), 67), (('it', "'", 's'), 67), ((',', 'and', 'then'), 67)]
"""
```
![Trigram](/images/trigrams-freqdist.png)

Filtramos los resultados para evitar conectores y signos de puntuación.
```py
threshold = 2
filtered_trigrams = [trigram for trigram in md_trigrams if len(trigram[0])>threshold and len(trigram[1])>threshold and len(trigram[2])>threshold]
filtered_dist_trigrams = FreqDist(filtered_trigrams)
filtered_dist_trigrams.most_common(20)
"""
[(('the', 'Sperm', 'Whale'), 77),
 (('the', 'White', 'Whale'), 63),
 (('the', 'old', 'man'), 32),
 (('the', 'sperm', 'whale'), 30),
 (('the', 'Right', 'Whale'), 25),
 (('the', 'same', 'time'), 24),
 (('for', 'the', 'time'), 24),
 (('must', 'have', 'been'), 23),
 (('into', 'the', 'sea'), 21),
 (('now', 'and', 'then'), 20),
 (('into', 'the', 'air'), 19),
 (('down', 'into', 'the'), 18),
 (('the', 'white', 'whale'), 18),
 (('the', 'Pequod', 'was'), 17),
 (('over', 'the', 'side'), 17),
 (('from', 'the', 'whale'), 16),
 (('all', 'the', 'time'), 16),
 (('round', 'and', 'round'), 16),
 (('from', 'the', 'ship'), 15),
 (('and', 'all', 'the'), 14)]
"""
```
Graficamos con `filtered_dist_trigrams.plot(20)`.
![Filtered trigram](/images/filtered-trigram-freqdist.png)

## Colocaciones en Python

**Colocaciones**: Son secuencias de palabras que ocurren en textos y conversaciones con una frecuencia inusualmente alta. Existe evidencia estadística de que estas palabras ocurren con esa frecuencia inusualmente alta, y esto nos da la idea de que podemos construir algunos números, algunas métricas que nos permiten identificar de manera sistemática estas colocaciones.

[NLTK Documentation](https://github.com/nltk/nltk/wiki#documentation)

[NLTK Book](/files/nltk-doc.pdf)

Para poder tener un indicio de esto, debemos usar el Pointwise Mutual Information (PMI) - Información Mutua Punto a Punto.

![Fórmula PMI](/images/formula-pmi.png)

Dado que el resultado de la división nos va a dar un número más pequeño que 1, la mayoría de los resultados nos va a dar negativos. Con el PMI buscamos los resultados más altos, o sea, los más cercanos a cero.

En este sentido, los bigramas que tengan más posibilidades de ser colocaciones deben tener valores del PMI cercanos a cero y por lo tanto tener los valores menos negativos posibles o al menos es lo que podemos ver en este momento.

## Colocaciones en gráficos de dispersión

Nos encontramos con un pequeño obstaculo. Hay PMIs cercanos a cero que la frecuencia del bigrama es 1, o sea que aparecen una sola vez en el texto, esto no es bueno, porque podemos encontrar bigramas con más frecuencia de repetición. Para poder identificar colocaciones de lenguaje no solo debemos considerar la métrica del PMI, sino también la frecuencia del n-grama en sí mismo, para lo cual nos sugiere que debemos considerar 2 métricas, la frecuencia del bigrama en sí y la métrica PMI.

**Creamos una nueva columna que contenga el logaritmo de la frecuencia de los bigramas.**
```py
df['PMI'] = df[['bigram_freq','word_0_freq','word_1_freq']].apply(lambda x: np.log2(x.values[0]/(x.values[1]*x.values[2])), axis = 1)
df['log_bigram_freq'] = df['bigram_freq'].apply(lambda x: np.log2(x))
df
"""
    bi-grams 	            word_0 	    word_1 	        bigram_freq 	word_0_freq 	word_1_freq 	PMI 	        log_bigram_freq
0 	(and, prophesies) 	    and 	    prophesies      1 	            6024 	        1 	            -12.556506 	    0.000000
1 	(lesson, which) 	    lesson 	    which 	        1 	            12 	            640 	        -12.906891 	    0.000000
2 	(not, unshunned) 	    not 	    unshunned 	    1 	            1103 	        1 	            -10.107217 	    0.000000
3 	(this, soliloquizer) 	this 	    soliloquizer 	1 	            1280 	        1 	            -10.321928 	    0.000000
4 	(the, precise) 	        the      	precise 	    11 	            13721 	        19 	            -14.532594 	    3.459432
"""
```

En el DF aplicamos el logaritmo sobre la frecuencia de aparición de la frecuencia de los bigramas, ¿Por qué hacemos esto? Porque el PMI en sí mismo ya es un valor que es el resultado de un logaritmo, y para que las 2 variables tengan la misma escala en el gráfico y el gráfico no se vea distorcionado es conveniente también aplicar logaritmo sobre la variable.

### Librería Plotly interactiva

Esta librería es extremadamente poderosa, pues, ofrece la posibilidad de crear gráficos interáctivos, en este caso, la librería nos ofrece la oportunidad de identificar las colocaciones dentro del libro Moby Dick.
```py
fig = px.scatter(x = df['PMI'].values, 
                 y = df['log_bigram_freq'], 
                 color = df['PMI']+df['log_bigram_freq'], 
                 hover_name = df['bi-grams'].values, 
                 width = 600, 
                 height = 600, 
                 labels= {'x':'PMI', 'y':'log (Bigram frequencies)'})
fig.show()
```

![Plotly](/images/plotly.png)

Podemos ver que la colocación más evidente es Moby Dick, y tiene total sentido, pues es el título del libro y uno de los personajes importantes, y más específicamente porque es un bigrama con una frecuencia de uso inusualmente alta en el lenguaje general. Las colocaciones sirven para identificar entonces, personas, lugares importantes para este caso literario identificar nombres propios, objetos y de esta manera empezar a asignar cierto tipo de etiquetas a palabras o expresiones que nos pueden dar información de elementos cruciales en el entendimiento del texto en sí mismo.

## Filtros y colocaciones en NLTK

### Medidas pre-construidas en NLTK

* [Source code for nltk.metrics.association](http://www.nltk.org/_modules/nltk/metrics/association.html)

Importamos una nueva librería y utilizamos nuevos métodos.
* `nltk.collocations.BigramAssocMeasures()`: Este método permite usar las métricas, incluyendo la PMI que estuvimos trabajando.
* `BigramCollocationFinder.from_words(text1)`: Método que permite que las palabras del texto 1 implemente una clase que nos va a ayudar a encontrar las colocaciones.
```py
from nltk.collocations import *
bigram_measure = nltk.collocations.BigramAssocMeasures() 
finder = BigramCollocationFinder.from_words(text1) 
```

Ahora, encontraremos las colocaciones usando NLTK.
```py
finder.apply_freq_filter(20)
finder.nbest(bigram_measure.pmi, 10)
"""
[('Moby', 'Dick'),
 ('Sperm', 'Whale'),
 ('White', 'Whale'),
 ('Right', 'Whale'),
 ('Captain', 'Peleg'),
 (',"', 'said'),
 ('never', 'mind'),
 ('!"', 'cried'),
 ('no', 'means'),
 ('each', 'other')]
"""
```

### Textos en español

Para trabajar con colocaciones en español es el mismo procedimiento que del inglés.
```py
nltk.download('cess_esp')
corpus = nltk.corpus.cess_esp.sents()
flatten_corpus = [w for l in corpus for w in l]
finder = BigramCollocationFinder.from_documents(corpus)
finder.apply_freq_filter(10)
finder.nbest(bigram_measure.pmi, 10)
"""
[('señora', 'Aguirre'),
 ('secretario', 'general'),
 ('elecciones', 'generales'),
 ('campaña', 'electoral'),
 ('quiere', 'decir'),
 ('Se', 'trata'),
 ('segunda', 'vuelta'),
 ('director', 'general'),
 ('primer', 'ministro'),
 ('primer', 'lugar')]
"""
```

[Archivo Colab](https://colab.research.google.com/drive/1rc7KrNYglV14L0cVCLAfkl2HKCcZIfyt?usp=sharing)

[Archivo local](/code/03_lenguaje_con_estadistica.ipynb)

## Introducción a los recursos léxicos

**¿Qué es un recurso léxico y como podemos usardo para nuestro procesamiento del leguaje en cuanto las tareas que se deben ejecutar?**

Es una colección de palabras o frases que puede o no contener meta datos o información acerca de los elementos de esa colección.

**¿Por qué es tan importante esto?**

En lenguajes como el español hay palabras que pueden tener diferentes significados, que dependiendo del contexto en el cual esa palabra esta siendo usada y esta información se puede categorizar y estructurar dentro de lo que llamamos un recurso léxico.

### ¿Cómo es?

* Calle [verbo] conjugación del verbo callar. *Le puedes decir que se **calle** o me va a enloquecer...*
* Calle [sustantivo] referencia al espacio público por donde hay transito. *Ten cuidado al cruzar la **calle** porque el semáforo no funciona...*

La **categoría léxica** es determinar el uso o calificativo de la palabra en cuestión, si es verbo, sustantivo, adjetivo, adverbio. Y por último, el **significado** o **descripción** respecto al uso específico de esa palabra en cada uno de los casos.

## Recursos léxicos en NLTK

>Son colecciones de palabras o frases que tienen asociadas etiquetas o meta-informacion de algún tipo (POS tags, significados gramaticales, etc ...)

**comentario**: POS (Part of Speech), también llamado etiquetado gramatical o etiquetado de palabras por categorias, consiste en etiquetar la categoria gramatical a la que pertence cada palabra en un volumen de texto, siendo las categorias:

* Sustantivos
* Adjetivos
* Articulos
* Pronombres
* Verbos
* Adverbios
* Interjecciones
* Preposiciones
* Conjunciones

En esta ocasión empezamos a usar el método stopwords de NLTK, stopwords es una lista de palabras inútiles del lenguaje, las cuales hay que filtrar para hacer un análisis NLP mucho más preciso y sin tanto ruido.

```py
# Vocabulario: Palabras únicas en un corpus
vocab = sorted(set(text1))
print(vocab[:20])
"""
['!', '!"', '!"--', "!'", '!\'"', '!)', '!)"', '!*', '!--', '!--"', "!--'", '"', '"\'', '"--', '"...', '";', '$', '&', "'", "',"]

"""

# Distribuciones: frecuencia de aparición
word_freq = FreqDist(text1)
# En cierto sentido, la distribución de frecuencia es un lexicón, porque la llave es la palabra y el valor es la información que se tiene de esa palabra.
# Y es un lexicón enriquecido, porque contiene información de la palabra, no solo es la colección de la palabra.
print(word_freq)
# <FreqDist with 19317 samples and 260819 outcomes>

# Stopwords: Palabras muy usadas en el lenguaje que usualmente son filtradas en un pipeline de NLP (useless words)
stopwords.words('spanish')[:20]
"""
['de',
 'la',
 'que',
 'el',
 'en',
 'y',
 'a',
 'los',
 'del',
 'se',
 'las',
 'por',
 'un',
 'para',
 'con',
 'no',
 'una',
 'su',
 'al',
 'lo']
"""
```

Para el procesamiento de texto, es relevante eliminar la cantidad de stopwords en el contenido del texto, si un texto tiene demasiadas stopwords, a lo mejor no es tan grande como pensaba que lo era inicialmente.
```py
def stopwords_percentage(text, lang):
    stopwd = stopwords.words(lang)
    content = [w for w in text if w.lower() not in stopwd]
    return f'{round(len(content)/len(text)*100, 2)}%'

# Analizando el libro de Moby Dick
stopwords_percentage(text1, 'english')
# '58.63%'
```

## NLTK para traducción de palabras

Para esta clase usaremos la clase Swadesh. Swadesh sirve para comparar palabras y realizar traducciones.

Importamos la librería.
```py
from nltk.corpus import swadesh

print(swadesh.fileids()) # lista de abreviaciones de lenguajes disponibles
# ['be', 'bg', 'bs', 'ca', 'cs', 'cu', 'de', 'en', 'es', 'fr', 'hr', 'it', 'la', 'mk', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'uk']

print(swadesh.words('en')[:20])
# ['I', 'you (singular), thou', 'he', 'we', 'you (plural)', 'they', 'this', 'that', 'here', 'there', 'who', 'what', 'where', 'when', 'how', 'not', 'all', 'many', 'some', 'few']
```

### Realizando traducciones con Swadesh
```py
fr2es = swadesh.entries(['fr','es'])
print(fr2es[:10])
# [('je', 'yo'), ('tu, vous', 'tú, usted'), ('il', 'él'), ('nous', 'nosotros'), ('vous', 'vosotros, ustedes'), ('ils, elles', 'ellos, ellas'), ('ceci', 'este'), ('cela', 'ese, aquel'), ('ici', 'aquí, acá'), ('là', 'ahí, allí, allá')]

translate = dict(fr2es)
translate['chien']
# 'perro'

translate['jeter']
# tirar
```

**Traduciendo del inglés**
```py
en2es = swadesh.entries(['en','es'])
translate = dict(en2es)
translate['dog']
# perro
```

Ya con esto tenemos la capacidad de usar recursos léxicos que nos permiten traducir de un idioma específico a otro. **¿Cómo podemos usar esto de forma más interesante?** Podemos conectar esto con otros conceptos de clases previas, imagina que estamos haciendo procesamiento de texto en un idioma diferente al propio, y lo que se quiere es generar tokenizadores de esos textos, tokenizar las palabras, filtrar stopwords y luego con las palabras relevantes, saber cuáles son las más mencionadas.

[Archivo Colab](https://colab.research.google.com/drive/1N-2fp7ku1V68kEMbpFU-CDyCjcqAPyNF?usp=sharing)

[Archivo local](/code/04_recursos_lexicos.ipynb)