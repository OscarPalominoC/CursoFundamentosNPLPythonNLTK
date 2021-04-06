"""
Una función que recibe un arreglo de palabras y una palabra debe retornar la cantidad de veces que aparece esa palabra en el arreglo y el largo del arreglo
Al llamarla con (["ab","a"], "a") debería retornar (1,2)
"""

def main(data):
    """
    Esta función recibe una tupla, dicha tupla contiene un array y una cadena, el objetivo de esta función es analizar cuántas veces se repite la cadena dentro de los elementos del array y también devolvernos la cantidad de elementos del array.

    Args:
        data (tupla): Contiene como primer elemento un array de elementos de tipo cadena, y el segundo elemento es una cadena.

    Returns:
        tupla: Esta tupla contiene la dimensión del array y la cantidad de veces que se repite la cadena dentro de los elementos del array.
    """
    array = data[0]
    word = data[1]
    result = 0
    for  text in array:
        result += text.count(word)

    return (len(array)-1, result)


if __name__ == '__main__':
    """
    El siguiente código recibe del usuario una oración y un carácter o serie de caracteres, para que posteriormente sean procesados por la función main().
    """
    sentence = input('Escribe la oración a analizar: ')
    word = input('Qué caracter o palabra quieres encontrar? ')
    array = sentence.split(' ')
    data = (array, word)
    main(data)