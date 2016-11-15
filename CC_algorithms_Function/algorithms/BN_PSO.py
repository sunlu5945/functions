'''

@author: sunlu
'''
import pandas as pd
from pgmpy.estimators import BdeuScore,K2Score,BicScore
from pgmpy.models import BayesianModel
import numpy as np
import random
import sys
import math
import numpy.random


#ccpso for function named SCHWEFEL from http://www.sfu.ca/~ssurjano/schwef.html
#best fitness:-418.9829*dimension
#best x:420.9687

class cc_frame:
    def __init__(self, pop_size,max_generation,groupsize_set,dimension,uppon,lower,pro):
        self.w = 0.7
        self.c1 = 1.57
        self.c2 = 1.57
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.group_size_set = groupsize_set
        self.dimension = dimension
        self.uppon = uppon
        self.lower = lower
        self.p = pro
        self.position = []
        self.velocity = []
        self.p_best = []
        self.g_best = []
        self.group = []
        self.l_best = []
        self.l_best_value = []
        self.p_best_value = []
        self.g_best_value = sys.maxint
        self.data = []

    def cal_fit(self,position):
        temp_fitness_value = 0.0
        for i in range(self.dimension):
            temp_fitness_value = temp_fitness_value + (position[i]) * (math.sin(math.sqrt(abs(position[i]))))  
        return -temp_fitness_value
        
    def init_g_best(self):
        for i in range(self.pop_size):
            self.p_best_value[i] = self.cal_fit(self.p_best[i])
        self.g_best_value = min(self.p_best_value)
        self.best_index =  self.p_best_value.index(self.g_best_value)
        for i in range(self.dimension):
            self.g_best.append(self.p_best[self.best_index][i])
    
    def do_grouping(self):
        #global group_size
        for i in range (self.group_size):
            self.group.append([])
        # random grouping
        dimension_list = []
        for i in range (self.dimension):
            dimension_list.append(i)
        random.shuffle(dimension_list)
        # do grouping
        sub_dim = self.dimension / self.group_size
        remain = self.dimension % self.group_size
        if remain == 0:
            for i in range(self.group_size):
                for j in range(sub_dim):
                    self.group[i].append(dimension_list[i*sub_dim+j])  
        if remain != 0:    
            for i in range(self.group_size):
                for j in range(sub_dim):
                    self.group[i].append(dimension_list[i*sub_dim+j])  
            for i in range(remain):
                self.group[self.group_size-1].append(dimension_list[self.group_size*sub_dim-1+i])
    
    def adjust_grouping(self):   
        for i in range(self.group_size): 
            test_bn = BN_learning(1,len(self.group[i]))  
    def best_among_local(self,temp_local_best_list):
        local_best_value = min(temp_local_best_list)
        #print 'min:', local_best_value
        best_index =  self.p_best_value.index(local_best_value)
        return best_index
    
    def find_local_best(self):
        temp_local_best_list = []
        for i in range(self.pop_size):
            if i == 0:
                temp_local_best_list.append(self.p_best_value[i+self.pop_size-1])
                temp_local_best_list.append(self.p_best_value[i])
                temp_local_best_list.append(self.p_best_value[i+1])
                #print 'i=0:', temp_local_best_list
            elif i+1 == self.pop_size:
                temp_local_best_list.append(self.p_best_value[i-1])
                temp_local_best_list.append(self.p_best_value[i])
                temp_local_best_list.append(self.p_best_value[(i+1) % self.pop_size])
                #print 'i=pop_size-1:', temp_local_best_list
                
            else:
                temp_local_best_list.append(self.p_best_value[i-1])
                temp_local_best_list.append(self.p_best_value[i])
                temp_local_best_list.append(self.p_best_value[i+1])
                #print 'i= others:', temp_local_best_list
            
            local_best_index = self.best_among_local(temp_local_best_list)
            #print '%s local_best_index:' %i, local_best_index
            self.l_best_value.append(self.p_best_value[local_best_index])
            del temp_local_best_list[:]
            for j in range(self.dimension):
                self.l_best.append(self.p_best[local_best_index])
                
    def cal_function_value(self,group_size_index,reference_particle,g_best):
        temp_position = []
        for i in range(len(g_best)):
            temp_position.append(g_best[i])
        for i in range(len(self.group[group_size_index])):
            dim_index = self.group[group_size_index][i]
            temp_position[dim_index] = reference_particle[dim_index]
        temp_function_value =  self.cal_fit(temp_position)
        return temp_function_value
    
    def replace(self,group_size_index,ref_particle,replaced_particle):   
        temp_particle = replaced_particle 
        for i in range(len(self.group[group_size_index])):
            dim_index = self.group[group_size_index][i]
            temp_particle[dim_index] = ref_particle[dim_index]
        return temp_particle
    
    def get_data(self,chromosome_bef,chromosome_cur):
        data = []
        for i in range(self.dimension):
            if chromosome_bef[i] < chromosome_cur[i]:
                data.append(1)
            elif chromosome_bef[i] >= chromosome_cur[i]:
                data.append(0)
        return data
    def update_velocity(self):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                self.velocity[i][j] = self.w*self.velocity[i][j]+self.c1*self.random.random()*(self.p_best[i][j]-self.position[i][j])+self.c2*random.random()*(self.g_best[j]-self.position[i][j])  
                
    def update_position(self):
        record_position = self.position
        for i in range(self.pop_size):
            for j in range(self.dimension):
                if random.random <= self.p:
                    self.position[i][j] = self.p_best[i][j]+numpy.random.standard_cauchy([1])*abs(self.p_best[i][j]-self.l_best[i][j])
                else:
                    self.position[i][j] = self.l_best[i][j]+np.random.normal(0,1)*abs(self.p_best[i][j]-self.l_best[i][j])
                #position[i][j] = position[i][j]+velocity[i][j]
                if abs(self.position[i][j]) > 500:
                    self.position[i][j] = random.uniform(self.uppon,self.lower)
        for i in range(self.pop_size):
            temp_data = self.get_data(record_position[i], self.position[i])
            self.data.append(temp_data)
                    
    def run_ccpso(self):
        #init_grouping
        random_index = random.randint(0,len(self.group_size_set)-1)
        self.group_size = self.group_size_set[random_index]
        
        for i in range(self.pop_size):
            self.position.append([])
            self.velocity.append([])
    
        for i in range(self.pop_size):
            self.p_best_value.append(0.0)
            for j in range(self.dimension):
                self.position[i].append(random.uniform(self.uppon,self.lower))
                self.velocity[i].append(random.uniform(self.uppon,self.lower))
                
        for i in range(self.pop_size):
            self.p_best.append(self.position[i])
            
        self.init_g_best()
        self.do_grouping()
        self.find_local_best()
        for i in range (self.max_generation):
            current_best = self.g_best_value 
            for j in range(self.group_size):
                for k in range(self.pop_size):
                    temp_value_1 = self.cal_function_value(j,self.position[k],self.g_best)
                    temp_value_2 = self.cal_function_value(j, self.p_best[k],self.g_best)
                    if temp_value_1 < temp_value_2:
                        self.p_best_value[k] = temp_value_1
                        self.p_best[k] = self.replace(j,self.position[k],self.p_best[k])
                    if temp_value_2 < self.g_best_value:
                        self.g_best_value = temp_value_2
                        self.g_best = self.replace(j,self.p_best[k],self.g_best)
            self.find_local_best()
            self.update_position()
            
            if (i+1) % 50== 0:
                for k in range(self.pop_size / 5 +1):
                    for j in range(self.dimension):
                        self.position[k][j] = random.uniform(self.uppon,self.lower)
            
            if(self.g_best_value >= current_best):
                random_index = random.randint(0,len(self.group_size_set)-1)
                print random_index
                self.group_size = self.group_size_set[random_index]
                self.do_grouping()
                print ",", self.group_size
            #print g_best
            print '%sth' %i, self.g_best_value
            
            
class BN_learning:
    def __init__(self,pop_size,node_num):
        self.pop_size = pop_size  # size of population used for BN structure learning
        self.population = []
        self.node_num = node_num
        self.dimension = (self.node_num * (self.node_num+1)/2)
        self.uppon = 5
        self.lower = -5
        #initialize population
        for i in range(self.pop_size):
            self.population.append([])
            self.population[i].append(1.5)
            self.population[i].append(2.3)
            self.population[i].append(0.8)
            self.population[i].append(3.2)
            self.population[i].append(2.9)
            self.population[i].append(1.8)
            self.population[i].append(2.1)
            self.population[i].append(1.1)
            self.population[i].append(2.2)
            self.population[i].append(2.3)
            #for j in range(self.dimension):
                #self.population[i].append(random.uniform(self.uppon,self.lower))
    #chromosome as parameter
    def decoding(self):
        node_seq = []
        edge_seq = []
        node_index = []
        matrix = []
        graph = []
        model_a = BayesianModel()
        
        for i in range(self.node_num):
            matrix.append([])
            for j in range(self.node_num):
                matrix[i].append(999)
        for i in range(self.node_num):
            node_seq.append(self.population[0][i])
        for i in range((self.node_num * (self.node_num-1))/2):
            edge_seq.append(int((round(abs(self.population[0][self.node_num+i])))%2))
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
            model_a.add_node(str(node_index[i]))
            for j in range(self.node_num-1-i):
                if matrix[i][j+1] == 1:
                    graph[i].append(node_index[j+1])    
        data = pd.DataFrame(np.random.randint(0,2,size=(100,4)),columns=list('1234'))
        for i in range(self.node_num):
            temp_in = node_index[i]
            for j in range(self.node_num-1-i):
                if matrix[i][j+1] == 1:
                    temp_out = node_index[j+1]          
                    model_a.add_edge(str(temp_in), str(temp_out))
        bdeu = BdeuScore(data,equivalent_sample_size=5)
        return bdeu
        
    def crossover(self):
        temp_fitness = []  # it contains 2*n  n: fitnesses   2n:crossover new fitnesses
        temp_population = []  # it contains 2*n  n: chromosomes   2n:crossover new chromosomes
        new_chromosome_1 = []  # first new chromosome of two chromosomes
        new_chromosome_2 = []  # second new chromosome of two chromosomes
        for i in range(self.pop_size):
            temp_population.append(self.population[i])
            temp_fitness.append(self.cal_fit(self.chromosome[i]))
        for i in range(self.pop_size):
            temp_cro_pro = random.uniform(0,1)
            if temp_cro_pro > self.cro_pro:
                cro_pos_index = random.randint(0,self.pop_size-1)
                cro_index = random.randint(0,self.dimension-1)
                for j in range(cro_index+1):
                    new_chromosome_1.append(self.chromosome[i][j])
                    new_chromosome_2.append(self.chromosome[cro_pos_index][j])
                for j in range(self.dimension-cro_index-1):
                    new_chromosome_1.append(self.chromosome[cro_pos_index][j+cro_index+1])
                    new_chromosome_2.append(self.chromosome[i][j+cro_index+1])
                new_fitness_1 = self.cal_fit(new_chromosome_1)
                new_fitness_2 = self.cal_fit(new_chromosome_2)
                for pi in range(self.pop_size):
                    if new_fitness_1 < temp_fitness[pi]:
                        for dim in range(self.dimension):
                            temp_population[pi][dim] = new_chromosome_1[dim]
                    if new_fitness_2 < temp_fitness[pi]:
                        for dim in range(self.dimension):
                            temp_population[pi][dim] = new_chromosome_2[dim]
        self.position = temp_population
        
    def mutation(self):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                temp_pro = random.uniform(0,1)
                if temp_pro < self.mut_pro:
                    self.temp_population[i][j] = random.uniform(self.lower,self.uppon)
    
    
    
    
test_ccpso = cc_frame(1000,1000,[2,5,50,100],100,500,-500,0.7)
#test_ccpso.test_print()
test_ccpso.run_ccpso()