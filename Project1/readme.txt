>Para execução (indexer.py tem tudo) existem alguns parâmetros:
	Obrigatórios:
		'-f FILENAME' -> Ficheiro a examinar
	Opcionais:	
		'-m NUMBER' ->  Dimensão mínima dos tokens
		'-ps' 	-> Porter Stemmer é utilizado
		'-s'	-> Usa as stopwords do ficheiro stopwords.txt
				>Cada linha deste ficheiro é interpretada como uma stopword
				>Para adicionar ou retirar stopwords basta eliminar linhas ou adicionar novas ao ficheiro
		'-c'	-> Utiliza as técnicas de compressão (id_stream, string_compression)

	EXEMPLO:	python3 indexer.py -f amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv.gz -m 3 -ps -s -c

> Quanto maior for o valor de block_size e MAX_ID_COUNTER definidas no inicio de indexer.py, menor são o numero de blocos intermedios
	> O nosso número é bastante baixo e por isso temos bastantes blocos, isto deve-se aos erros de memória que ocorriam sem motivo aparente
		(suspeitamos que tenha sido por causa das definições do pycharm). Por outro lado, utilizando o VSCode surgiam alguns problemas de
		imports que também não conseguimos solucionar 
		
>O índice é um simples dicionário com o mapeamento =>	termo : string_concatenada_com_ids_respetivos

>Blocos são escritos sempre que a id_stream tem 150 mil ids ou quando o tamanho do indice é superior a 750000 bytes

>A Compressão dos termos (string_compression) é básica
	Exemplo: AAABBC -> A3B2C

>Para compressão, em vez de serem repetidas as Ids de cada review, as mesmas são guardadas numa string e no índice estão guardadas ponteiros para a id 
na string
	>Como existem muitas repetiçoes de ids no índice, a utilização dos ponteiros permite reduzir quase para metade a memória utilizada pelo índice
	>Como esta id_stream fica muito grande e nenhuma id se repete, sempre que um bloco é guardado da-se append da id_stream desse bloco
		no ficheiro id_stream.txt
  