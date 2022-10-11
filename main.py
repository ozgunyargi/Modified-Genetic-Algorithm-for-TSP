import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from config import *
from GeneticAlgo import City, Fitness
import random


def k_means_population(cities, pop_size=5, n_cluster=3):

    kmeans = KMeans(n_clusters= n_cluster, random_state=42)
    coordinates = np.array([[x.lat, x.lon] for x in cities])
    kmeans.fit(coordinates)
    labels = list(set(kmeans.labels_))
    cluster_dict = {}

    for city in cities:
        label = kmeans.predict([[city.lat, city.lon]])[0]
        if label not in cluster_dict:
            cluster_dict[label] = [city]
        else:
            cluster_dict[label].append(city)

    cluster_centers = kmeans.cluster_centers_

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
                cities_org = cluster_dict[cluster]
                cities_ = cities_org.copy()
                while len(cities_) != 0:
                    city = random.sample(cities_, 1)[0]
                    inner_route.append(city)
                    cities_.pop(cities_.index(city))
                chromosome += inner_route
            if chromosome not in population:
                population.append(chromosome)
                in_population=False
    return population


def create_initial_population(cities, pop_size = 10):
    population = []

    for i in range(pop_size):
        route = random.sample(cities, len(cities))
        while route in population:
            route = random.sample(cities, len(cities))
        population.append(route)

    return population

def find_cumulative_probabilities (mydict):

    fitness_scores = [mydict[route]["fitness"] for route in mydict]

    max_score = max(fitness_scores)
    min_score = min(fitness_scores)

    cumulative_dict = {}
    for key in mydict:
        try:
            cumulative_probability = (mydict[key]["fitness"]-min_score)/(max_score-min_score)
        except:
            sorted_ = {k: v for k, v in sorted(mydict.items(), key=lambda item: item[1]["fitness"])}
            print([x[1]["fitness"] for x in sorted_.items()])
            print([x.name for x in sorted_[list(sorted_.keys())[-1]]["route"]])
            print([x.name for x in sorted_[list(sorted_.keys())[0]]["route"]])
            exit(0)
        cumulative_dict[key] = cumulative_probability
    return cumulative_dict

def next_population(mydict, populationDict, top=2, Pc=0.4, Pm=0.05):
    next_population_dict = {}
    num_of_iteration = len(mydict)
    temp_dict = mydict.copy()
    currIndx = list(populationDict.keys())[-1]

    if currIndx == 0:
        for i in range(top):
            temp_dict.pop(list(mydict.keys())[(i+1)*-1])

    else:
        chosenChromosomes = []

        for i in range(top):
            tempIndx = currIndx
            Temperature = 300
            step_size = 0.1
            chosen_chromosome = mydict[list(mydict.keys())[(i+1)*-1]]

            while Temperature > 0.01:
                for i in range(top):
                    if tempIndx == 0:
                        break

                    previous_indx = random.randint(0, top-1)
                    previous_chromosome = populationDict[tempIndx-1][previous_indx]

                    delta_c = chosen_chromosome["fitness"] - previous_chromosome["fitness"]

                    if delta_c < 0:
                        chosen_chromosome = previous_chromosome
                        tempIndx -= 1
                    else:
                        pickProb = random.uniform(0,1)
                        if pickProb < np.exp(-1*delta_c/Temperature):
                            chosen_chromosome = previous_chromosome
                            tempIndx -= 1

                Temperature = Temperature*step_size

                if tempIndx == 0:
                    break
            chosenChromosomes.append(chosen_chromosome)

    cumulatives = find_cumulative_probabilities(temp_dict)

    for i in range(num_of_iteration):
        in_population = True
        if i <top:
            if currIndx == 0:
                next_population_dict[i] = mydict[list(mydict.keys())[(i+1)*-1]]
            else:
                next_population_dict[i] = chosenChromosomes[i]
        else:
            temp_route = find_route(cumulatives) # Roulette Wheel Approach
            th_c = random.uniform(0,1)

            if th_c > Pc: # Check for crossover
                th_m = random.uniform(0,1)
                if th_m > Pm: # Check for mutation
                    next_population_dict[i] = mydict[temp_route]
                else:
                    next_population_dict[i] = {"route": mutation(temp_dict[temp_route]), "fitness": 0}
            else:
                temp_route2 = find_route(cumulatives)
                while temp_route2 == temp_route:
                    temp_route2 = find_route(cumulatives)
                crossover_route = crossover(temp_dict[temp_route], temp_dict[temp_route2])
                th_m = random.uniform(0,1)
                if th_m > Pm: # Check for mutation
                    next_population_dict[i] = crossover_route
                else:
                    next_population_dict[i] = {"route": mutation(crossover_route), "fitness": 0}

    return next_population_dict

def crossover (parent1, parent2):

    uniform_weight = 0
    method_picker = random.uniform(0,1)

    # Uniform Crossover method
    if method_picker <= uniform_weight:

        cut1 = random.randint(1,len(parent1["route"])-2)
        cut2 = random.randint(1,len(parent1["route"])-2)
        while cut1==cut2:
            cut2 = random.randint(1,len(parent1["route"])-2)
        if cut2>cut1:
            lower=cut1
            upper=cut2
        else:
            lower=cut2
            upper=cut1

        parent1_cut = parent1["route"][lower:upper+1]
        parent2_cut = parent2["route"][lower:upper+1]

        offspring = []
        filled = False

        for indx, element in enumerate(parent1["route"]):
            if indx<lower or indx>upper:
                if element not in parent2_cut:
                    offspring.append(element)
                else:
                    replaced_element = random.choice(parent1_cut)
                    while replaced_element in parent2_cut:
                        replaced_element = random.choice(parent1_cut)
                    offspring.append(replaced_element)
                    parent1_cut.pop(parent1_cut.index(replaced_element))
            elif filled == False:
                offspring += parent2_cut
                filled = True
        return {"route":offspring, "fitness": parent1["fitness"]}

    # Adapted Crossover Method (from paper)
    else:

        parent1FirstTwoDist = parent1["route"][0].distance(parent1["route"][1])
        parent2FirstTwoDist = parent2["route"][0].distance(parent2["route"][1])

        if parent1FirstTwoDist <= parent2FirstTwoDist:
            offspring = parent1["route"][:2] + parent2["route"][2:]
            for indx in range(2, len(offspring)):
                if offspring.count(offspring[indx]) != 1:
                    for city in parent1["route"]:
                        if city not in offspring:
                            offspring[indx] = city
                            break
            return {"route":offspring, "fitness": parent1["fitness"]}

        else:
            offspring = parent2["route"][:2] + parent1["route"][2:]
            for indx in range(2, len(offspring)):
                if offspring.count(offspring[indx]) != 1:
                    for city in parent2["route"]:
                        if city not in offspring:
                            offspring[indx] = city
                            break
            return {"route":offspring, "fitness": parent2["fitness"]}

def mutation (route):
    dict_lenght = len(route["route"])
    swap_weight = 0
    method_picker = random.uniform(0,1)

    # Swap Mutation
    if method_picker <= swap_weight:
        first_indx = random.randint(0, dict_lenght-1)
        second_indx = random.randint(0, dict_lenght-1)
        while second_indx == first_indx:
            second_indx = random.randint(0, dict_lenght-1)

        mutated_list = []

        for indx,city in enumerate(route["route"]):
            if indx == first_indx:
                mutated_list.append(route["route"][second_indx])
            elif indx == second_indx:
                mutated_list.append(route["route"][first_indx])
            else:
                mutated_list.append(city)
        return mutated_list

    # Translation Mutation
    else:
        picked_route = random.randint(0, dict_lenght-1)
        translated_loc = random.randint(0, dict_lenght-1)
        while translated_loc == picked_route:
            translated_loc = random.randint(0, dict_lenght-1)

        mutated_list = []
        counter = 0
        for indx, city in enumerate(route["route"]):
            if indx == translated_loc:
                mutated_list.append(route["route"][picked_route])
                mutated_list.append(city)
            elif indx == picked_route:
                continue
            else:
                mutated_list.append(city)

        return mutated_list

def find_route(mydict):
    roulette_weight = 0.4
    tournament_weight = roulette_weight+ 0.4
    method_picker = random.uniform(0,1)

    # Roulette wheel
    if method_picker <= roulette_weight:
        cum_prob_th = random.uniform(0,1)
        for route in mydict:
            if cum_prob_th < mydict[route]:
                return route

    # Tournament
    elif method_picker <= tournament_weight:
        tournament_pop_size=3
        participants = random.sample(list(mydict.keys()), tournament_pop_size)
        random_pick = np.Inf
        while random_pick > max([mydict[indx] for indx in participants]):
            random_pick = random.uniform(0,1)
        for route in sorted(participants):
            if random_pick <= mydict[route]:
                return route
    # Random
    else:
        return random.sample(list(mydict.keys()), 1)[0]

def main():

    counter = 0
    populationElitistDict = {}

    #PARAMETERS
    iter_num = 300
    crossover_prob = 0.5
    mutation_prob = 0.1
    elitist = 20
    population_size = 500

    # CREATE INITIAL POPULATION
    df = pd.read_csv(DATA)
    mycities = df.apply(lambda x: City(x["city"], x["longitude"], x["latitude"]), axis=1).values.tolist()
    population = k_means_population(mycities, pop_size=population_size, n_cluster=3)
#    population = create_initial_population(mycities, population_size)

    route_dict = {}
    for indx, route in enumerate(population):
        route_fitness = Fitness(route).calculate_fitness()
        route_dict[indx] = {"route": route, "fitness": route_fitness}

    sortedRoutesByFitness = {k: v for k, v in sorted(route_dict.items(), key=lambda item: item[1]["fitness"])}
    populationElitistDict[counter] = [sortedRoutesByFitness[list(sortedRoutesByFitness.keys())[key]] for key in range(-1, (elitist+1)*-1, -1)]
    best_route = populationElitistDict[counter][-1]

    counter +=1

    distance_scores = [1/sortedRoutesByFitness[list(sortedRoutesByFitness.keys())[-1]]["fitness"]]

    # GENERATE NEXT POPULATIONS
    for indx, i in enumerate(tqdm(range(iter_num))):            # 0,0-1,0-2,1

      new_pop = next_population(sortedRoutesByFitness, populationElitistDict, top=elitist, Pc=crossover_prob, Pm=mutation_prob)
      routes = [x[1]["route"] for x in new_pop.items()]

      route_dict = {}
      for indx,route in enumerate(routes):
          route_fitness = Fitness(route).calculate_fitness()
          route_dict[indx] = {"route": route, "fitness": route_fitness}
      sortedRoutesByFitness = {k: v for k, v in sorted(route_dict.items(), key=lambda item: item[1]["fitness"])}
      populationElitistDict[counter] = [sortedRoutesByFitness[list(sortedRoutesByFitness.keys())[key]] for key in range((elitist+1)*-1, -1)]
      distance_scores.append(1/sortedRoutesByFitness[list(sortedRoutesByFitness.keys())[-1]]["fitness"])

      if populationElitistDict[counter][-1]["fitness"] > best_route["fitness"]:
          best_route = populationElitistDict[counter][-1]

      counter += 1

    #PLOT RESULTS
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    x = np.array([i for i in range(iter_num+1)])
    X_Y_Spline = make_interp_spline(x, distance_scores)
    X_ = np.linspace(x.min(), x.max(), 100)
    Y_ = X_Y_Spline(X_)

    ax[0].plot(x, distance_scores, alpha=0.8)
    ax[0].set_xlabel("Number of iterations")
    ax[0].set_ylabel("Distance to take")
    ax[0].set_title("GA Observation Plot")
    ax[0].text(len(distance_scores)//2, distance_scores[0], f"Min Dist: {1/best_route['fitness']}", horizontalalignment="center", verticalalignment="top")

    optimum_route = best_route

    for city_indx in range(len(optimum_route["route"])-1):
        city = optimum_route["route"][city_indx]
        next_city = optimum_route["route"][city_indx+1]
        ax[1].scatter(city.lat, city.lon, c="blue", alpha=0.6)
        ax[1].text(city.lat, city.lon+0.1, city.name, ha="center", va="top", fontsize=6)
        ax[1].plot([city.lat, next_city.lat], [city.lon, next_city.lon], color="red", alpha=0.3)
    ax[1].scatter(optimum_route["route"][-1].lat,optimum_route["route"][-1].lon, c="blue", alpha=0.6)
    ax[1].text(optimum_route["route"][-1].lat,optimum_route["route"][-1].lon+0.1, optimum_route["route"][-1].name, ha="center", va="top", fontsize=6)
    ax[1].plot([optimum_route["route"][0].lat, optimum_route["route"][-1].lat], [optimum_route["route"][0].lon, optimum_route["route"][-1].lon], color="red", alpha=0.3)
    ax[1].set_title("City Locations")

    plt.tight_layout()
    plt.savefig("Results/TSP_GA_Analysis_ElitistWithSA.png")
    plt.show()

if __name__ == '__main__':
    random.seed(42)
    main()
