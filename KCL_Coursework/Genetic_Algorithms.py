import random

# Example from the web. In the module we built out in the pacman environment. This is neater
# https://medium.com/@Data_Aficionado_1083/genetic-algorithms-optimizing-success-through-evolutionary-computing-f4e7d452084f

def initialize_pop(TARGET):
    population = list()
    tar_len = len(TARGET)
    for i in range(POP_SIZE):
        temp = list()
        for j in range(tar_len):
            temp.append(random.choice(GENES))
        population.append(temp)
    return population

def crossover(selected_chromo, CHROMO_LEN, population):
    offspring_cross = []
    for i in range(int(POP_SIZE)):
        parent1 = random.choice(selected_chromo)
        parent2 = random.choice(population[:int(POP_SIZE * 50)])
        p1 = parent1[0]
        p2 = parent2[0]
        crossover_point = random.randint(1, CHROMO_LEN - 1)
        child = p1[:crossover_point] + p2[crossover_point:]
        offspring_cross.extend([child])
    return offspring_cross

def mutate(offspring, MUT_RATE):
    mutated_offspring = []
    for arr in offspring:
        for i in range(len(arr)):
            if random.random() < MUT_RATE:
                arr[i] = random.choice(GENES)
        mutated_offspring.append(arr)
    return mutated_offspring

def selection(population, TARGET):
    sorted_chromo_pop = sorted(population, key=lambda x: x[1])
    return sorted_chromo_pop[:int(0.5 * POP_SIZE)]

def fitness_cal(TARGET, chromo_from_pop):
    difference = 0
    for tar_char, chromo_char in zip(TARGET, chromo_from_pop):
        if tar_char != chromo_char:
            difference += 1
    return [chromo_from_pop, difference]

def replace(new_gen, population):
    for _ in range(len(population)):
        if population[_][1] > new_gen[_][1]:
            population[_][0] = new_gen[_][0]
            population[_][1] = new_gen[_][1]
    return population

POP_SIZE = 500
MUT_RATE = 0.1
TARGET = 'patrick cresswell'
GENES = ' abcdefghijklmnopqrstuvwxyz'
# Initialize population
initial_population = initialize_pop(TARGET)
found = False
population = []
generation = 1
# Calculate the fitness for the current population
for _ in range(len(initial_population)):
    population.append(fitness_cal(TARGET, initial_population[_]))
# Loop until TARGET is found
while not found:
    # Select best candidates from current population
    selected = selection(population, TARGET)
    # Crossover to make new generation
    population = sorted(population, key=lambda x: x[1])
    crossovered = crossover(selected, len(TARGET), population)
    # Apply mutation to diversify the new generation
    mutated = mutate(crossovered, MUT_RATE)
    new_gen = []
    for _ in mutated:
        new_gen.append(fitness_cal(TARGET, _))
    # Replacement of least fit of old generation with new generation
    population = replace(new_gen, population)
    print('Best fit in new population: ' + str(population[0][0]) + ' Generation: ' + str(generation) + ' Fitness: ' + str(population[0][1]))
    if (population[0][1] == 0): break
    generation += 1