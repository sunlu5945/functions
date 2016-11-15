'''

@author: sunlu
'''
import pandas as pd
from pgmpy.estimators import BdeuScore,K2Score,BicScore
from pgmpy.models import BayesianModel
import numpy as np
import random
import sys

'''
    implemented according to Carvalho A. A cooperative co-evolutionary genetic algorithm for learning bayesian network structures[C]
    Proceedings of the 13th annual conference on Genetic and evolutionary computation. ACM, 2011: 1131-1138.
'''
class BN_learning:
    def __init__(self,pop_size,node_num,max_generation,cro_pro,mut_pro,current_group):
        #pso parameters
        self.pop_size = pop_size  # size of population used for BN structure learning
        self.max_generation = max_generation
        self.cro_pro = cro_pro
        self.mut_pro = mut_pro
        #pso poluation
        self.population = []
        self.node_num = node_num
        self.dimension = (self.node_num * (self.node_num+1)/2)
        self.uppon = 5
        self.lower = -5
        self.p_best_value = []
        self.p_best = []
        self.g_best = []
        n = np.random.randint(0,2,size=(100,4))
        self.data = pd.DataFrame(n,columns=list('1234'))
        self.best_bn = BayesianModel()
        self.current_group = current_group
    def decoding(self,chromosome):
        node_seq = []
        edge_seq = []
        node_index = []
        matrix = []
        graph = []
        model_bn = BayesianModel()
        for i in range(self.node_num):
            matrix.append([])
            for j in range(self.node_num):
                matrix[i].append(999)
        for i in range(self.node_num):
            node_seq.append(chromosome[i])
        for i in range((self.node_num * (self.node_num-1))/2):
            edge_seq.append(int(round(chromosome[self.node_num+i]) % 2))
        sorted_seq = sorted(node_seq,reverse=True)
        for i in range(self.node_num):
            node_index.append(node_seq.index(sorted_seq[i])+1)
        count = 0  
        
        
        for i in range(self.node_num):
            for j in range(self.node_num-1-i):
                matrix[i][j+i+1] = edge_seq[count]
                count = count + 1
        for i in range(self.node_num):
            graph.append([])
            graph[i].append(node_index[i])
            model_bn.add_node(str(node_index[i]))
            for j in range(self.node_num-1-i):
                if matrix[i][j+1] == 1:
                    graph[i].append(node_index[j+1])
        for i in range(self.node_num):
            temp_in = node_index[i]
            for j in range(self.node_num-1-i):
                if matrix[i][j+1] == 1:
                    temp_out = node_index[j+1]          
                    model_bn.add_edge(str(temp_in), str(temp_out))
        return model_bn
    def find_single_node(self,bn):
        node_in_bn = self.best_bn.nodes(data=False)
        print node_in_bn
        degree = bn.degree()
        print degree
        key_list = []
        value_list = []
        for key,value in degree.items():  
            key_list.append(key)  
            value_list.append(value)  
        get_value = 0
        if get_value in value_list:  
            get_value_index = value_list.index(get_value)  
            print 'The single node is:',key_list[get_value_index]
            return get_value_index
        else:
            print 'There are no single nodes'
    #chromosome as parameter
    def cal_fit(self,chromosome):
        model_a = self.decoding(chromosome)
        bdeu = BdeuScore(self.data,equivalent_sample_size=5)
        return bdeu.score(model_a)
    
    def init_population(self):
        # fixed encoding used for test
        #for i in range(self.pop_size):
            #self.population.append([])
            #self.population[i].append(1.5+i)
            #self.population[i].append(2.3+i)
            #self.population[i].append(0.8+i)
            #self.population[i].append(3.2+i)
            #self.population[i].append(2.9+i)
            #self.population[i].append(1.8+i)
            #self.population[i].append(2.1+i)
            #self.population[i].append(1.1+i)
            #self.population[i].append(2.2+i)
            #self.population[i].append(2.3+i)
        #print 'init_population:', self.population
        
        #initialize population
        for i in range(self.pop_size):
            self.population.append([])
            for j in range(self.dimension):
                self.population[i].append(random.uniform(self.uppon,self.lower))
    def init_p_best(self):
        for i in range(self.pop_size):
            self.p_best.append(self.population[i])
            self.p_best_value.append(self.cal_fit(self.population[i]))
        print 'init_p_best:',self.p_best
        print 'init_p_best_value', self.p_best_value
    def init_g_best(self):# BDe max
        self.g_best_value = max(self.p_best_value)
        self.best_index =  self.p_best_value.index(self.g_best_value)
        for i in range(self.dimension):
            self.g_best.append(self.p_best[self.best_index][i])
        print 'init_g_best',self.g_best
        print 'init_g_best_value',self.g_best_value
    def crossover(self):
        temp_fitness = []  # it contains 2*n  n: fitnesses   2n:crossover new fitnesses
        temp_population = []  # it contains 2*n  n: chromosomes   2n:crossover new chromosomes
        new_chromosome_1 = []  # first new chromosome of two chromosomes
        new_chromosome_2 = []  # second new chromosome of two chromosomes
        for i in range(self.pop_size):
            temp_population.append(self.population[i])
            temp_fitness.append(self.cal_fit(self.population[i]))
        for i in range(self.pop_size):
            temp_cro_pro = random.uniform(0,1)
            if temp_cro_pro < self.cro_pro:
                cro_pos_index = random.randint(0,self.pop_size-1) # the 2nd chromosome used for crossover
                print 'cro_pos_index:',cro_pos_index
                cro_index = random.randint(0,self.dimension-1)
                print 'cro_index',cro_index
                for j in range(cro_index+1):
                    new_chromosome_1.append(self.population[i][j])
                    new_chromosome_2.append(self.population[cro_pos_index][j])
                for j in range(self.dimension-cro_index-1):
                    new_chromosome_1.append(self.population[cro_pos_index][j+cro_index+1])
                    new_chromosome_2.append(self.population[i][j+cro_index+1])
                new_fitness_1 = self.cal_fit(new_chromosome_1)
                new_fitness_2 = self.cal_fit(new_chromosome_2)
                print 'new_chromosome_1:',new_chromosome_1,'new_fitness_1:',new_fitness_1
                print 'new_chromosome_2:',new_chromosome_2,'new_fitness_2',new_fitness_2
                
                for pi in range(self.pop_size):
                    if new_fitness_1 > temp_fitness[pi]:
                        for dim in range(self.dimension):
                            temp_population[pi][dim] = new_chromosome_1[dim]
                    if new_fitness_2 > temp_fitness[pi]:
                        for dim in range(self.dimension):
                            temp_population[pi][dim] = new_chromosome_2[dim]
            del new_chromosome_1[:]
            del new_chromosome_2[:]
        self.population = temp_population
        print 'after crossover:',self.population

    def mutation(self):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                temp_pro = random.uniform(0,1)
                if temp_pro < self.mut_pro:
                    self.population[i][j] = random.uniform(self.lower,self.uppon)
        print 'after mutation:',self.population
        
    def run_learning(self):
        self.init_population()
        self.init_p_best()
        self.init_g_best()
        for i in range (self.max_generation):
            for k in range(self.pop_size):
                temp_value = self.cal_fit(self.population[k])
                if temp_value > self.p_best_value[k]:
                        self.p_best_value[k] = temp_value
                        for dim in range(self.dimension):
                            self.p_best[k][dim] = self.population[k][dim]
                        if self.p_best_value[k] > self.g_best_value:
                            self.g_best_value = self.p_best_value[k]
                            for dim in range(self.dimension):
                                self.g_best[dim] = self.p_best[k]
            self.crossover()
            self.mutation()
    #def adjust_grouping(self):
    def get_best_bn(self):
        self.best_bn = self.decoding(self.g_best)
        print 'model',self.best_bn
        return self.best_bn
    
    def adjust_grouping(self,best_bn,group):
        single_node = self.find_single_node(best_bn)
        print single_node
        new_group = []
        single_node_list = []
        for m in range(len(group)):
            new_group.append([])
            for n in range(len(group[m])):
                if group[m][n] != single_node:
                    new_group[m].append(group[m][n])
                else:
                    single_node_list.append(group[m][n])
        print single_node_list
        for l in range(len(single_node_list)):
            rand_group_index = random.randint(0,len(group)-1)
        
        new_group[rand_group_index].append(single_node_list[l])
        
        return new_group

group = [[0,1,3,5],[2,4,6,7]]
for i in range(2):
    test_bn = BN_learning(2,len(group[i]),1,0.7,0.3,group[i])
    test_bn.run_learning()
    bn_best = BayesianModel()
    bn_best = test_bn.get_best_bn()
    new_group = test_bn.adjust_grouping(bn_best,group)
    print new_group
    