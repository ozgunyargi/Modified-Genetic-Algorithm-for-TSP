import pandas as pd
import numpy as np
import random, os, glob
import matplotlib.pyplot as plt
from config import *
from GeneticAlgo import City, Fitness
from sklearn.cluster import KMeans
from scipy.interpolate import make_interp_spline


def elbow(K, df):

    distortions = []

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        distortions.append(kmeans.inertia_)

    X_Y_Spline = make_interp_spline(K, distortions)
    X_ = np.linspace(min(K), max(K), max(K)-min(K)+1)
    Y_ = X_Y_Spline(X_)

    plt.figure(figsize=(8,8))
    plt.plot(X_, Y_)
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")
    plt.show()

def create_initial_population(centers, df, pop_size = 500):
    cluster_centers = centers.cluster_centers_
    labels = list(set(centers.labels_))

    chosen_cluster = random.sample(labels, 1)[0]
    greedy_path = [chosen_cluster]

    while len(labels) != 1:

        min_dist = np.Inf
        for i in labels:
            if i != chosen_cluster:
                dist = ( (cluster_centers[chosen_cluster,0]-cluster_centers[i,0])**2+(cluster_centers[chosen_cluster,1]-cluster_centers[i,1])**2 )**(0.5)
                if dist < min_dist:
                    next_cluster = i
                    min_dist = dist
        greedy_path.append(next_cluster)
        labels.pop(labels.index(next_cluster))

    population = []

    for i in range(pop_size):
        in_population = True
        while in_population:
            chromosome = []
            for cluster in greedy_path:
                inner_route = []
                cities = df[df["groups"] == cluster]["city"].values.tolist()
                while len(cities) != 0:
                    city = random.sample(cities, 1)[0]
                    inner_route.append(city)
                    cities.pop(cities.index(city))
                chromosome += inner_route
            if chromosome not in population:
                population.append(chromosome)
                in_population=False

def main():
    df_ = pd.read_csv(DATA)
    elbow(range(1,20), df_)
    exit(0)
    n_cluster = 3

    kmeans = KMeans(n_clusters= n_cluster, random_state=42)
    coordinates = df_[["latitude", "longitude"]].to_numpy()

    kmeans.fit(coordinates)
    df_["groups"] = df_.apply(lambda x: kmeans.predict([[x["latitude"], x["longitude"]]]), axis=1)

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    for group in range(n_cluster):

        df_temp = df_[df_["groups"] == group]

        lats = df_temp["latitude"].values
        lons = df_temp["longitude"].values
        names = df_temp["city"].values

        ax.scatter(lats,lons)
        for indx, name in enumerate(names):
            ax.text(lats[indx], lons[indx]+0.1, name, fontsize=6, ha="center", va="top")

    ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
    for i in range(n_cluster):
        ax.text(kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1]+0.1, i, fontsize=6, ha="center", va="top")

    create_initial_population(kmeans, df_)
    plt.show()

if __name__ == '__main__':
    main()

    files = glob.glob("./Datasets/*.tsp")

    for file in files:
        rows = []
        file_name = os.path.basename(file)[:-4]
        print(file_name)
        with open(file) as myfile:
            lines = [line.strip() for line in myfile]
        start_index = lines.index("NODE_COORD_SECTION")+1
        stop_indx = lines.index("EOF")
        coordinates = lines[start_index:stop_indx]
        for coordinate in coordinates:
            splitted_row = coordinate.replace("  ", " ").strip().split(" ")
            city = splitted_row[0].strip()
            lat = float(splitted_row[1].strip())
            lon = float(splitted_row[2].strip())
            rows.append([city, lat, lon])
        rows = np.array(rows)
        df = pd.DataFrame(rows, columns = ["city", "latitude", "longitude"])
        df.to_csv(f"./Datasets/{file_name}.csv")
