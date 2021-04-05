# Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK

![Logo](https://static.platzi.com/media/achievements/badge-fundamentos-procesamiento-lenguaje-natural-python-85faceb3-d6e5-4829-9c80-847.png)

## Profesor

Francisco Camacho

## Archivos

[Slides](/files/slides.pdf)

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

