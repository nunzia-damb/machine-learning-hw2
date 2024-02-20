def skf(results, keyword):   
	
	'''Safe Key Finder: recebe um dicionario e uma key, ou lista
	para encontrar de forma encadeada, quando convertido do json e 
	procura de forma tolerante a erros, devolve str nula se nao encontra'''
	
	for i_key in keyword:
		if isinstance(results, dict):
			if i_key in results:
				results = results[i_key]			
			else:
				return ''
		elif isinstance(results, list) and isinstance(i_key, int):
			if len(results)>i_key:
				results = results[i_key]
			else:
				return ''
		else:
			return ''
	return results