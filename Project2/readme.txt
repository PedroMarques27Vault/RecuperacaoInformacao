>Para execução (indexer.py tem tudo) existem alguns parâmetros:
	Obrigatórios:
		'-f FILENAME' -> Ficheiro a examinar
	Opcionais:
	    '-i'        -> O índice é criado antes de o o search se realizar
	    '-w RANKING' -> Tipo de ranking, DEFAULT: lnc.ltc
		'-m NUMBER' ->  Dimensão mínima dos tokens
		'-ps' 	-> Porter Stemmer é utilizado
		'-s'	-> Usa as stopwords do ficheiro stopwords.txt
				>Cada linha deste ficheiro é interpretada como uma stopword
				>Para adicionar ou retirar stopwords basta eliminar linhas ou adicionar novas ao ficheiro
		'-c'	-> Utiliza as técnicas de compressão (id_stream, string_compression)
		'-k1 FLOAT' ->DEFAULT: 1.2, parametro k1 para bm25
		'-b FLOAT'  -> DEFAULT: 0.75, parametro b para bm25

	EXEMPLO:	python3 indexer.py -f amazon_reviews_us_Digital_Video_Games_v1_00.tsv.gz -i -m 5 -s -ps -c -w lnc.ltc

> Quanto maior for o valor de block_size definido no inicio de indexer.py, menor são o numero de blocos intermedios


>O índice (index.txt) é um simples dicionário com o mapeamento =>	termo = id1:w1|id2:w2...
    >Este peso não está normalizado

>idf.txt contém os IDF para cada termo
>settings.txt contém informações sobre como foi processada a indexação
>doc_len.txt contém valores que permitem calcular a dimensão de cada documento, para ajudar na normalizacao de pesos
    >Pode ser a dimensão do documento para bm25
    >caso contrario, dimensao_documento = square_root(valor_no_ficheiro)

>term_index_mapper permite mapear palavras com a sua localização no index
	EXEMPLO:
		aba-bota: 20
		bala-cola: 55
		>Isto significa que todas as palavras X tal que X=>aba e X<=bota estao no chunk entre 0 e 20 do ficheiro index.txt
		> Todas as palavras Y tal que Y=>bala e Y<=cola estao no chunk entre 21 e 55 do ficheiro index.txt

>A Compressão dos termos (string_compression) é básica
	Exemplo: AAABBC -> A3B2C

>Para compressão, em vez de serem repetidas as Ids de cada review, as mesmas são guardadas numa string e no índice estão guardadas ponteiros para a id 
na string
	>Como existem muitas repetiçoes de ids no índice, a utilização dos ponteiros permite reduzir quase para metade a memória utilizada pelo índice
	>Como esta id_stream fica muito grande e nenhuma id se repete, sempre que um bloco é guardado da-se append da id_stream desse bloco
		no ficheiro id_stream.txt
  