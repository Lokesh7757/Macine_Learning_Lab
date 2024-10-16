import numpy as np
import matplotlib.pyplot as plt

points = np.array([[0.1, 0.6], [0.2, 0.3], [0.15, 0.71], [0.08, 0.9], [0.25, 0.5], [0.24, 0.1], [0.16, 0.85], [0.3, 0.2]])
print("Given Dataset :")
print(points)

plt.figure(figsize=(5,5))
plt.scatter(x=points[:,0],y = points[:,1])
plt.title("Initially")
plt.show()

k = 2
c1 = points[0]
c2 = points[1]

plt.figure(figsize=(5,5))
plt.scatter(x=points[:,0],y = points[:,1])
plt.scatter([c1[0],c2[0]],[c1[1],c2[1]],marker="*")
plt.title("Step 1")
plt.show()

def manhattan(p1, p2):
    return np.abs((p1[0]-p2[0])) + np.abs((p1[1]-p2[1]))

def get_costs(data, medoids):
    clusters = {i:[] for i in range(len(medoids))}
    cst = 0
    for d in data:
        dst = np.array([manhattan(d, md) for md in medoids])
        c = dst.argmin()
        clusters[c].append(d)
        cst+=dst.min()

    clusters = {k:np.array(v) for k,v in clusters.items()}
    return clusters, cst

def KMedoids(data, k, iters = 100):
    medoids = np.array([data[i] for i in range(k)])
    samples,_ = data.shape

    #print("\ndata:",data)

    clusters, cost = get_costs(data, medoids)
    count = 0

    print("\nBefore :")
    print("\nClusters 1:\n",clusters[0])
    print("\nClusters 2:\n",clusters[1])
    #print("\nCost :",cost)

    colors =  np.array(np.random.randint(0, 255, size =(k, 4)))/255
    colors[:,3]=1

    plt.title(f"Step : 0")
    [plt.scatter(clusters[t][:, 0], clusters[t][:, 1], marker="*", s=100, color = colors[t]) for t in range(k)]
    plt.scatter(medoids[:, 0], medoids[:, 1], s=200, color=colors)
    plt.show()

    while True:
        swap = False
        for i in range(samples):
            if not i in medoids:
                for j in range(k):
                    tmp_meds = medoids.copy()
                    tmp_meds[j] = i
                    clusters_, cost_ = get_costs(data, tmp_meds)

                    if cost_<cost:
                        medoids = tmp_meds
                        cost = cost_
                        swap = True
                        clusters = clusters_
                        print("\nAfter :")
                        print("\nClusters 1:\n",clusters[0])
                        print("\nClusters 2:\n",clusters[1])
                        print(f"\nMedoids Changed to:\n {medoids}")
                        plt.title(f"Step : {count+1}")  
                        [plt.scatter(clusters[t][:, 0], clusters[t][:, 1], marker="*", s=100, color = colors[t]) for t in range(k)]
                        plt.scatter(medoids[:, 0], medoids[:, 1], s=200, color=colors)
                        plt.show()
        count+=1
        print("\nAfter :")
        print("\nClusters 1:\n",clusters[0])
        print("\nClusters 2:\n",clusters[1])
        if count>=iters:
            print("End of the iterations.")
            break
        if not swap:
            print("\nNo changes.")
            break

KMedoids(points,2) 
