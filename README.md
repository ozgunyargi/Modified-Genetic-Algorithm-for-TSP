# Modified-Genetic-Algorithm-for-TSP
This repository simulates an experiment on genetic algorithm application in TSP with modifications.

## GENERATE INITIAL POPULATION

- Kmeans algorithm is used
	- Hyperparameter "number of cluster" is determined by using Elbow method for data specific example. It is chosen as 3.
	- Flow of the operation:
		- Cities are clustered and the closest clusters were determined by using Euclidean distance where the cluster
		  centers were used to calculate the distance between each cluster. The sequence that takes a minimum distance to travel around
		  clusters was found using a greedy search.
		- To create a chromosome, cities belonging to same cluster are randomly picked as each city will appear only one time
		  inside of the sequence. Then, according to the cluster sequence that we have found in the previous operation, randomly picked
		  city sequences are merged as they appear in the same order with cluster sequence.
		- To create a population, iterate this operation many times. If newly generated chromosome is already inside of the population,
		  ignore newly create chromosome and move on to the next operation.


## GENERATE NEXT POPULATIONS

- Elitism is used to generate first chromosomes for next population
- For newly generated chromosomes
	- Calcultate cumulative probabilities for each chromosome from the previous population by looking the fitness score.
		- Fitness score is calculated as the following: 1/(Distance to visit all cities in the given order by using Haversine distance)
			- Since we want to find the minimum distance, we divide 1 with found distance to convert task to maximization problem. 
	- Random number is generated to determine selection type. Three different approaches were used.
		- Roulette wheel with prob 0.4 => Generate random number. Pick the chromosome that has higher cumulative probability than the number as well as the closest to it.
		- Tournament with prob 0.4 => Pick more than one chromosome from the population randomly. The chromosome that has the highest fitness score will be picked.
		- Random pick with prob 0.2
	- For chosen chromosome:
		- Check if there is a crossover by generating another random number.
			- **If** generated number is smaller than or equal to crossover threshold (which was set as 0.8) do crossover.
				- After crossover check if there is a mutation by generating another number and compare with mutation threshold (0.25)
				- **If** number is smaller than or equal to mutation threshold, do mutation and transfer to next population
				- **Else**, transfer crossover chromosome to next population
			- **Else**, check if there is a mutation by generating another random number and compare it with mutation threshold (0.25)
				- **If** there is no mutation, directly transfer chosen chromosome to next population
				- **Else**, do mutation and transfer to next population



## CROSSOVER
- Two method was implamented: 
  - Uniform crossover, 
  - A method implamented from the paper (Dwivedi, 2012)
###  METHODOLOGY
- Generate a random number
- **If** number is smaller than or equal to uniform crossover threshold, implament uniform crossover.
- **Else**, implament adopted new method.
	- Brief explanation of adopted method
		- Consider first two cities from both parents.
		- Calculate the distances between two cities (for the first parent, calculate the distance from city1 to city2
		  which city1 and city2 exist in the parent as the first two city. Do same calculation for second parent as well)
		- Find the parent that has smaller distance between first two city than the other. Pick those cities as the starting point of the new sequence.
		- For the rest of the sequence, use the other parent that have not chosen. Transfer the sequence in the same order from the unused parent
		  starting from the third city.
``` 
parent1= 1-2-3-4-5-6
parent2= 5-2-1-4-3-6
			
dist(1,2) = 5
dist(5,2) = 7
			
new sequence = 1-2-1-4-3-6
Since 1 occurs more than one in the new sequence, chance it with not existing city
ex: (continue)
new sequence = 1-2-1-4-3-6 => (unused cities = {5})
Adjusted sequence = 1-2-5-4-3-6 => valid solution

```
## MUTATION
- Two method was implamented:
  - Swap mutation,
  - Translation mutation
### METHODOLOGY
- Generate a random number
- **If** number is smaller than or equal to swap mutation threshold, implament swap mutation.
- **Else**, implament translation muation.

## References
- Dwivedi, V., Chauhan, T., Saxena, S., & Agrawal, P. (2012). Travelling salesman problem using genetic algorithm. IJCA proceedings on development of reliable information systems, techniques and related issues (DRISTI 2012), 1, 25.
