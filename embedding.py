def embedFlukes(flukes):
    embeddings = []
    for f in flukes:
        i = f[2] #flukes is structured [path,middlepoint,image]
        #do embedding
        embed = f[2]
        embeddings.append([i,embed])
    return embeddings

    