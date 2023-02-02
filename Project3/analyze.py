import math


def main():
    q = load_query('queries.relevance.txt')

    for top in [10,20,50]:
        for i in ['search_results/bm25.txt', 'search_results/bm25_boosted.txt',
                  'search_results/lnc.ltc.txt', 'search_results/lnc.ltc_boosted.txt']:
            name = f"Statistics{top}{i.split('/')[1]}"
            results = load_results(i)
            p_recall = get_precision_recall_f_measure(results, q, top)
            ndcg = get_ndcg(results, q, top)
            ap = get_average_precision(results, q, top)
            write_results(name, p_recall, ndcg, ap)


def write_results(name, p_recall, ndcg, ap):
    open(f"Statistics/{name}",'w')
    f = open(f"Statistics/{name}", "a+")
    for query in p_recall:

        f.write(f"\nQ:{query}\n")
        f.write(f"Precision: {p_recall[query][0]}\n")
        f.write(f"Recall: {p_recall[query][1]}\n")
        f.write(f"F_Measure: {p_recall[query][2]}\n")
        f.write(f"nDCG: {ndcg[query]}\n")
        f.write(f"Average Precision: {ap[query]}\n")


def get_average_precision(results, queries, top_n=50):
    final = dict()
    for query in queries:
        r = results[query][0][:top_n+1]
        q = queries[query][0][:top_n+1]
        relevance, pk = get_precision_at_k(r, q, top_n)
        if len([x for x in relevance if x != 0]) != 0:
            final[query] = sum([relevance[i] * pk[i] for i in range(len(relevance))]) / len([x for x in relevance if x != 0])
        else:
            final[query] = 0
    return final


def get_precision_at_k(r, q, top_n):
    relevance = []

    for i in range(top_n + 1):
        if r[i] in q:
            relevance.append(1)
        else:
            relevance.append(0)

    return relevance, [sum(relevance[:i + 1]) / (i + 1) for i in range(len(relevance))]


def get_ndcg(data_r, data_query, top_n=50):
    final = dict()

    for query in data_query:
        top_queries = [(data_query[query][0][x],int(data_query[query][1][x])) for x in range(top_n+1)]
        intersection = [x for x in top_queries if x[0] in data_r[query][0][:top_n+1]]
        ideal_dcg = top_queries[0][1] + sum([top_queries[i-1][1]/math.log2(i) for i in range(2, top_n)])

        if len(intersection) == 0:
            my_dcg = 0
        elif len(intersection) == 1:
            my_dcg = intersection[0][1]
        else:
            my_dcg = intersection[0][1] + sum([intersection[i-1][1]/math.log2(i) for i in range(2, len(intersection))])

        final[query] = my_dcg/ideal_dcg

    return final


def load_query(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    ids = {}
    current_list = [[], []]
    query = ""

    for i in range(len(data)):
        line = data[i]
        if "Q:" in line:
            if query != "":
                ids[query] = current_list
                query = line.split(':')[1].split('\n')[0].lower()
                current_list = [[], []]
            else:
                query = line.split(':')[1].split('\n')[0].lower()

        elif line != "\n":
            sep = line.split('\t')
            current_list[0].append(sep[0])
            current_list[1].append(sep[1].split('\n')[0])

    ids[query] = current_list
    return ids


def load_results(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    ids = {}
    current_list = [[], []]
    query = ""

    for line in data:
        if "Q:" in line:
            if query != "" or line == data[-1]:
                ids[query] = current_list
                query = line.split(':')[1][1:-1].lower()
                current_list = [[], []]
            else:
                query = line.split(':')[1][1:-1].lower()

        elif line != "\n":
            sep = line.split(' ')
            current_list[0].append(sep[0])
            current_list[1].append(sep[1].split('\n')[0])

    ids[query] = current_list
    return ids


def get_precision_recall_f_measure(query1, query2, size):
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F Measure = 2*P*R / (P+R)
    results = {}
    size += 1

    for i in query1:
        true_p = [x for x in query1[i][0] if x in query2[i.lower()][0][:size]]
        false_p = [x for x in query1[i][0][:size] if x not in query2[i.lower()][0][size:]]
        false_n = [x for x in query2[i.lower()][0][:size] if x not in query1[i][0][size:]]

        precision = len(true_p) / (len(true_p) + len(false_p))
        recall = len(true_p) / (len(true_p) + len(false_n))
        _sum = precision+recall

        if _sum == 0:
            _sum = 1

        f_measure = 2*precision*recall / _sum
        results[i] = [precision, recall, f_measure]

    return results


main()
