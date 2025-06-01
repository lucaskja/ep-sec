import re
import os
from unidecode import unidecode
import numpy as np
import string
import random
import math

def sortear_arquivo_txt(diretorio):
    arquivos = os.listdir(diretorio)
    arquivos_txt = [arquivo for arquivo in arquivos if arquivo.endswith('.txt')]
    
    if not arquivos_txt:
        return "Nenhum arquivo .txt encontrado no diretório."
    
    arquivo_sorteado = random.choice(arquivos_txt)
    
    return arquivo_sorteado

def parse(file_name):

	# abre o arquivo para leitura
	with open(file_name, 'r', encoding="iso-8859-1") as arquivo_entrada:
	    # lê o conteúdo do arquivo
	    conteudo = arquivo_entrada.read()

	# transforma as letras em minúsculas
	conteudo = conteudo.lower()

	# remove os acentos das vogais
	conteudo = unidecode(conteudo)

	# remove todos os caracteres que não são letras
	conteudo = re.sub(r'[^a-z]', '', conteudo)
	return conteudo

def save_file(file_name,conteudo):
	# abre um novo arquivo para escrita
	with open(file_name, 'w') as arquivo_saida:
	    # escreve o conteúdo no arquivo
	    arquivo_saida.write(conteudo)

def enc_monosyllabic(conteudo,tamanho,grupo):
	n = len(conteudo)-tamanho+1
	r = np.random.randint(0, n)
	texto_aberto = conteudo[r:r+tamanho]

	az = string.ascii_lowercase

	key = np.random.permutation(26)
	key = ''.join([az[key[i]] for i in range(26)])

	key_enc = {az[i] : key[i] for i in range(26)}
	key_dec = {key[i] : az[i] for i in range(26)}

	texto_cifrado = [key_enc[i] for i in texto_aberto]

	# texto_aberto = [key_dec[i] for i in texto_cifrado]

	save_file('Cifrado/Mono/' + grupo + '_' + 'texto_cifrado.txt',''.join(texto_cifrado))
	save_file('Aberto/Mono/' + grupo + '_' + 'key.txt',key)
	save_file('Aberto/Mono/' + grupo + '_' + 'texto_aberto.txt',''.join(texto_aberto))

def inv_multiplicativo(b,m):
  A = np.array([1, 0, m])
  B = np.array([0, 1, b])
  
  while True:
    if B[2] == 0:
      return 0
  
    if B[2] == 1:
      return B[1] % m 
  
    Q = np.floor( A[2]/B[2] )
    T = A - Q*B
    A = B
    B = T


def enc_hill(conteudo,tamanho,grupo,k):
	n = len(conteudo)-tamanho+1
	r = np.random.randint(0, n)
	texto_aberto = conteudo[r:r+tamanho]

	az = string.ascii_lowercase
	alf2dec = {az[i] : i for i in range(26)}
	dec2alf = {i : az[i] for i in range(26)}
	
	
	
	while True:
	# Gera uma matriz triangular superior
		#key = [[0] * k for _ in range(k)]
		key = np.zeros((k,k))
		for i in range(k):
			for j in range(i, k):
				if i == j:
					# Garante que os elementos da diagonal sejam coprimos com 26
					while True:
						element = random.randint(1, 25)
						if math.gcd(element, 26) == 1:
						    key[i][j] = element
						    break
				else:
					key[i][j] = random.randint(0, 25)

		# Calcula o determinante
		det = int(round(np.linalg.det(key)))

		# Verifica se o determinante é coprimo com 26
		if math.gcd(det, 26) == 1:
		    break
		else:
		    print('bug')	
	


	texto_numerico = [alf2dec[i] for i in texto_aberto]
	texto_cifrado = np.array(texto_numerico).reshape((int(tamanho/k),k)).T
	texto_cifrado = key@texto_cifrado % 26
	texto_cifrado = texto_cifrado.T.reshape(tamanho)
	texto_cifrado = [dec2alf[i] for i in texto_cifrado]



	save_file('Cifrado/Hill/' + grupo + '_' + str(k) + '_' +  'texto_cifrado.txt',''.join(texto_cifrado))
	save_file('Aberto/Hill/' + grupo + '_' + str(k) + '_' +  'key.txt',np.array2string(key))
	save_file('Aberto/Hill/' + grupo + '_' + str(k) + '_' +  'texto_aberto.txt',''.join(texto_aberto))


def enc_vigenere(conteudo,tamanho,grupo,k):
	n = len(conteudo)-tamanho+1

	r = np.random.randint(0, n)
	texto_aberto = conteudo[r:r+tamanho]

	key = np.random.randint(0, 26, (k))
	while key.size < tamanho:
		key = np.concatenate((key,key))
		
	key = key[0:tamanho]

	az = string.ascii_lowercase
	alf2dec = {az[i] : i for i in range(26)}
	dec2alf = {i : az[i] for i in range(26)}

	texto_numerico = [alf2dec[i] for i in texto_aberto]
	texto_cifrado = ( np.array(texto_numerico) + key ) % 26
	texto_cifrado = [dec2alf[i] for i in texto_cifrado]

	key_texto = [dec2alf[i] for i in key]


	save_file('Cifrado/Vigenere/' + grupo + '_' + str(k) + '_' +  'texto_cifrado.txt',''.join(texto_cifrado))
	save_file('Aberto/Vigenere/' + grupo + '_' + str(k) + '_' +  'key.txt',''.join(key_texto))
	save_file('Aberto/Vigenere/' + grupo + '_' + str(k) + '_' +  'texto_aberto.txt',''.join(texto_aberto))


def main():

	group = 'Grupo20'
	diretorio = 'textos'
	
	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	enc_monosyllabic(conteudo,120,group)


	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 2
	enc_hill(conteudo,120,group,k)

	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 3
	enc_hill(conteudo,120,group,k)

	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 4
	enc_hill(conteudo,120,group,k)

	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 5
	enc_hill(conteudo,120,group,k)

	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 20
	enc_vigenere(conteudo,120,group,k)

	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 30
	enc_vigenere(conteudo,120,group,k)

	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 40
	enc_vigenere(conteudo,120,group,k)
	
	arquivo_sorteado = sortear_arquivo_txt(diretorio)
	conteudo = parse(diretorio + '/' + arquivo_sorteado)
	k = 60
	enc_vigenere(conteudo,120,group,k)

if __name__ == "__main__":
    main()

