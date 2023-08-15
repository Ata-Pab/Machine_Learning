'''
OZYEGIN UNIVERSITY

CS 551 Introduction to Artificial Intelligence Assignment-2

Solving the Knapsack Problem via Genetic Algorithm

ATALAY PABUSCU
026217


Genetic Algorithm

The knapsack problem aims to maximize the total value for a selection of items, 
each with a value vi and a weight wi, from a given item set of size n, while 
simultaneously preventing the violation of the constraint that the total weight 
of selected items exceeds the knapsack’s capacity W.

Initialize populations P randomly
Determine fitness of population
Repeat until convergence:
    Select parents from population
    Crossover and generate new population
    Perform mutation on new population
    Calculate fitness for new population


Pseudocode of the Genetic Algorithm

t = 0
initialize(P(t=0))
evaluate(P(t=0))
while isNotTerminated() do:
    Pp(t) = P(t).select_parents()
    Pc(t) = reproduction(Pp)
    mutate(Pc(t))
    evaluate(Pc(t))
    P(t+1) = build_next_generation_from(Pc(t), P(t))
    t = t + 1
end



References
https://www.geeksforgeeks.org/genetic-algorithms/
http://www.ra.cs.uni-tuebingen.de/software/JCell/tutorial/ch03s05.html
https://arpitbhayani.me/blogs/genetic-knapsack
https://medium.com/koderunners/genetic-algorithm-part-3-knapsack-problem-b59035ddd1d6
'''

import pandas as pd
import random
import matplotlib.pyplot as plt

NUM_POPULATION  = 100
weight_limit = 0

def get_knapsack_data(loc):
    data = pd.read_excel(loc)
    if 'Weights' in data.columns:
        number_of_items = len(data['Weights'][:])
        if number_of_items == 0:
            raise ValueError("There is no item in the list")
        else:            
            return data
    else:
        raise NameError("There is no 'Weights' column in file")

def base_2(num):
    out = 1
    for i in range(num):
        out *= 2
    return out

class GeneticAlgorithm():
    def __init__(self, size, items, weights, values):
        self.size = size    # Number of individuals (chromosomes) in the population
        self.items = items  # Number of items (genes) in an individual (chromosome)
        self.weights = weights    # weights of each items
        self.values = values      # values of each items
        self.max_fitness = 0      # Maximum fitness score for given limit
        self.fitness_list = []    # Fitness scores for each individual (solution)
        self.weight_list = []     # Total weights for each individual (solution)
        self.gen_count = 0
        self.history_avg_fitness = []
        self.history_max_fitness = []
        self.history_best_ind = [] # Best fitted individual (solution) 
        # Check the population size control
        if base_2(items) < size:
            self.size = base_2(items)
            print(("UserWarning: Population size can not be bigger than possible outcomes\n \
            Population size was set to maximum value " + str(self.size)))
        self.ind_list = self.__create_population(self.size, self.items)

    def __create_population(self, num_ind, num_genes):
        ind_list = []
        ind_x = 0
        while ind_x < num_ind:
            ind = []
            for ix in range(num_genes):
                ind.append(random.randint(0, 1))
            if not ind in ind_list:
                ind_x += 1
                ind_list.append(ind)

        return ind_list

    # Calculate the fitness (returns the total value for selected solutions-individuals)
    # If the total weight is exceed the weight limits, returns 0 
    def __fitness_function(self):
        self.fitness_list = []   # Clear fitness_list list for new fitness function calculation
        self.weight_list = []    # Clear weight_list list for new fitness function calculation
        
        for ind_x in range(len(self.ind_list)):
            ind = self.ind_list[ind_x]
            self.fitness_list.append(0)
            self.weight_list.append(0)

            for i in range(self.items):
                self.fitness_list[ind_x] += self.values[i]*ind[i]
                self.weight_list[ind_x] += self.weights[i]*ind[i]
            # Return fitness score 0 if the weight limit is exceeded
            if self.weight_list[ind_x] > weight_limit:
                self.fitness_list[ind_x] = 0

    # Crossover is an evolutionary operation between two individuals, 
    # and it generates children having some parts from each parent.
    # It takes two different individual indexes as a parameter
    def crossover(self, ix1, ix2):
        cross_point = random.randint(0, (len(self.ind_list[ix1])-1)) # Radnom one-point crossover
        child1 = self.ind_list[ix1].copy()
        child2 = self.ind_list[ix2].copy()
        
        for ix in range(cross_point):
            child1[ix], child2[ix] = child2[ix], child1[ix]
        print("\nCross-point: ", cross_point)

        return (child1, child2)

    # The mutation is an evolutionary operation that randomly mutates an individual
    # It takes two different individual indexes as a parameter
    def mutation(self, child1, child2):
        mut_point = random.randint(0, (len(child1)-1))  # Random mutation point
        print("\nFirst mutation-point: ", mut_point)

        if child1[mut_point] == 0: child1[mut_point] = 1
        else: child1[mut_point] = 0

        mut_point = random.randint(0, (len(child1)-1))
        print("\nSecond mutation-point: ", mut_point)

        if child2[mut_point] == 0: child2[mut_point] = 1
        else: child2[mut_point] = 0

        return (child1, child2)

    # Selection process of the Genetic algorithm returns two
    # different individual indexes according to selection criteria
    def selection(self, method='best'):
        if method == 'best':
            return self.__get_fittest_indexes()
        elif method == 'tournament':
            return self.__get_tournament_indexes()
        else:
            raise ValueError("Invalid selection method type, use 'best' or 'tournament'")

    # Call this method after crossover and mutation phases of the algorithm
    def offspring(self, child1, child2):        
        min_ix1, min_ix2 = self.__get_least_fittest_indexes()

        # Replace least fittest individual with the fittest offspring
        self.ind_list[min_ix1] = child1
        self.ind_list[min_ix2] = child2

    def run_generations(self, generation, method='best', verbose=0):
        next_ind = []
        avg_fitness_scores = []
        self.history_avg_fitness = []
        self.history_max_fitness = []
        self.gen_count = 0
        max_fit = 0

        for _ in range(generation):
            self.__fitness_function()
            select_ix1, select_ix2 = self.selection(method)   # Select best fitting individuals
            
            self.history_avg_fitness.append(int(sum(self.fitness_list) / len(self.fitness_list)))
            if self.fitness_list[select_ix1] > max_fit: 
                max_fit = self.fitness_list[select_ix1]
                self.history_best_ind = self.ind_list[select_ix1]
            self.history_max_fitness.append(max_fit)
            
            if verbose > 0:
                self.__show_result_steps(select_ix1)
            
            child1, child2 = self.crossover(select_ix1, select_ix2)
            child1, child2 = self.mutation(child1, child2)
            # Calculate fitness function again for each individual after crossover and mutation phases
            # self.__fitness_function()
            self.offspring(child1, child2)
            self.gen_count += 1

    def __show_result_steps(self, ix1):
        print("\n", (self.gen_count + 1), ".", "Generated Population: ")
        print("",self.get_population())
        print("Fitness scores for each individual: ")
        print("",self.get_fitness_scores())
        print("Best fitting Individual and Fitness Score: ", self.ind_list[ix1], "  --  ", self.fitness_list[ix1])
        print("Average Fitness Score of the population: ", self.history_avg_fitness[self.gen_count])
        print("Maximum Fitness Score of the population: ", self.history_max_fitness[self.gen_count])

    def visualize_solution(self):
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.plot(list(range(self.gen_count)), self.history_avg_fitness, label = 'Average Fitness score')
        plt.plot(list(range(self.gen_count)), self.history_max_fitness, label = 'Maximum Fitness score')
        plt.legend()
        plt.title('Fitness Scores for all generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    def print_solution(self):
        print("Solution of the given Knapsack problem")
        print("Population size: ", self.size)
        print("Generation count: ", self.gen_count)
        print("Maximum Fitness Score: ", max(self.history_max_fitness))
        print("Maximum Average Fitness Score: ", max(self.history_avg_fitness))
        print("Best fitting Individual: ", self.history_best_ind)

    def __get_tournament_indexes(self):
        ix_shuffle = [*range(len(self.fitness_list))]
        random.shuffle(ix_shuffle)
        ix_1 = 0

        if self.fitness_list[ix_shuffle[0]] > self.fitness_list[ix_shuffle[1]]: ix_1 = ix_shuffle[0]
        else: ix_1 = ix_shuffle[1]

        if self.fitness_list[ix_shuffle[2]] > self.fitness_list[ix_shuffle[3]]:
            return ix_1, ix_shuffle[2]
        else:
            return ix_1, ix_shuffle[3]

    def __get_fittest_indexes(self):
        max_fit_ind1 = 0
        max_fit_ind2 = 0
        max_ix = -1

        for ind_x in range(len(self.ind_list)):
            if self.fitness_list[ind_x] > self.fitness_list[max_fit_ind1]:
                max_fit_ind2 = max_fit_ind1
                max_fit_ind1 = ind_x
            elif self.fitness_list[ind_x] > self.fitness_list[max_fit_ind2]:
                max_fit_ind2 = ind_x
        
        #if self.fitness_list[max_fit_ind1] == 0:
        #    return -1, -1
        #elif self.fitness_list[max_fit_ind2] == 0:
        #    return max_fit_ind1, -1

        return max_fit_ind1, max_fit_ind2

    def __get_least_fittest_indexes(self):
        min_fit_ind1 = 0
        min_fit_ind2 = 0

        for ind_x in range(len(self.ind_list)):
            if self.fitness_list[ind_x] < self.fitness_list[min_fit_ind1]:
                min_fit_ind2 = min_fit_ind1
                min_fit_ind1 = ind_x
            elif self.fitness_list[ind_x] < self.fitness_list[min_fit_ind2]:
                min_fit_ind2 = ind_x

        return min_fit_ind1, min_fit_ind2

    def get_population(self):
        return self.ind_list

    def get_fitness_scores(self):
        return self.fitness_list

    # Returns least fitted individual
    def get_least_fittest_individual(self):
        self.__fitness_function()
        min_ix1 = self.__get_least_fittest_index()
        return self.ind_list[min_ix1]

    # Returns fittest individual pairs
    def get_fittest_individuals(self):
        self.__fitness_function()

        max_ix1, max_ix2 = self.__get_fittest_indexes()
        
        if max_ix1 == -1:
            raise ValueError("There is no solution for this weight limit")
        elif max_ix2 == -1:
            print("There is only one solution for this weight limit")
            return self.ind_list[max_ix1], -1
        else:
            self.max_fitness = self.fitness_list[max_ix1]
            return self.ind_list[max_ix1], self.ind_list[max_ix2]


def main():
    global weight_limit
    data = get_knapsack_data("Knapsack.xlsx")

    weight_limit = int(input("Enter the weight limit: "))

    weights = list(data['Weights'][:])
    values = list(data['Values'][:])
    number_of_items = len(data['Weights'][:])

    print("Knapsack Data")
    print("Weight List: ", weights)
    print("Value List: ", values)
    print("Number of items: ", number_of_items)

    genetic_algorithm = GeneticAlgorithm(NUM_POPULATION, number_of_items, weights, values)

    genetic_algorithm.run_generations(generation=1000, method='tournament', verbose=0)
    genetic_algorithm.visualize_solution()
    genetic_algorithm.print_solution()


if __name__=="__main__":
    main()