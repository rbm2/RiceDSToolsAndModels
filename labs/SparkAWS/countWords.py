def countWords (sc, fileName):
    textfile = sc.textFile(fileName)
    lines = textfile.flatMap(lambda line: line.split(" "))
    counts = lines.map (lambda word: (word, 1))
    aggregatedCounts = counts.reduceByKey (lambda a, b: a + b)
    return aggregatedCounts.top (200, key=lambda p : p[1])
