'''

@author: sunlu
'''
import numpy as np
import random
import sys
import math
import numpy.random
from parameters import parameter as par
#ccpso for function named SCHWEFEL from http://www.sfu.ca/~ssurjano/schwef.html
#best fitness:-418.9829*dimension
#best x:420.9687

class ccpso:

    def __init__(self, pop_size,max_generation,groupsize_set,dimension,uppon,lower,pro,function_index):
        self.w = 0.7
        self.c1 = 1.57
        self.c2 = 1.57
        self.function_index = function_index
        
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

    #funtion 1 [-100,100] bias -450
    def sphere_function(self,position):
        res = 0
        for i in range(self.dimension):
            res +=  position[i] * position[i]
        res = res - 450
        return res
    #funciton 2 from Code  
    def elliptic_function(self,position):
        res = 0.0
        for i in range(self.dimension):
            res = math.pow(1e6, i/(self.dimension-1) * position[i] *position[i])
        return res
    #function 3 [-5,5] bias = -330  https://www.sfu.ca/~ssurjano/rastr.html
    def rastrigin_function(self,position):
        res = 0
        for i in range(self.dimension):
            res += position[i] * position[i] - 10 * math.cos(2 * math.pi * position[i]) + 10
        return res
    #function 4 [-32,32] bias = -140   https://www.sfu.ca/~ssurjano/ackley.html
    def ackley_function(self,position):
        res = 0
        for i in range(self.dimension):
            temp_gene = position[i]
            temp1 = 0
            temp2 = 0
            for i in range(self.dimension):
                temp1 += temp_gene * temp_gene
                temp2 += math.cos(2 * math.pi * temp_gene)
            temp1 = math.sqrt(temp1/self.dimension)
            temp2 = temp2 / self.dimension
            res = -20 * math.exp(-0.2 * temp1) - math.exp(temp2) + 20 + math.e
        return res
    #function 5 [-500,500] from http://www.sfu.ca/~ssurjano/schwef.html #best fitness:-418.9829*dimension #best x:420.9687
    def schwefel_function(self,position):
        res = 0
        for i in range(self.dimension):
            res +=  (position[i]) * (math.sin(math.sqrt(abs(position[i]))))
        return res
    #function 6 [-100,100] bias = 390
    def rosenbrock_function(self,position):
        res = 0;
        for i in range(self.dimension-1):
            temp_gene_1 = position[i]
            temp_gene_2 = position[i+1]
            res +=  100 * math.pow(temp_gene_1 * temp_gene_1 - temp_gene_2, 2) + math.pow(temp_gene_1 - 1, 2)
        return res
    #function 7  [-600,600] bias = -180
    def griewank_functopm (self,position):
        temp1 = 0
        temp2 = 0
        for i in range(self.dimension-1):
            j = i+1
            temp1 += (position[j] * position[j])/4000
            temp2 += math.cos(position[j] / math.sqrt(j))
        res = temp1 - temp2 +1
        return res
        
    def cal_fit(self,position):
        if self.function_index == 1:
            res = self.sphere_function(position)
            return res
        elif self.function_index == 2:
            res = self.elliptic_function(position)
            return res
        elif self.function_index == 3:
            res = self.rastrigin_function(position)
            return res
        elif self.function_index == 4:
            res = self.ackley_function(position)
            return res
        elif self.function_index == 5:
            res = self.schwefel_function(position)
            return res
        elif self.function_index == 6:
            res = self.rosenbrock_function(position)
            return res
        elif self.function_index == 7:
            res = self.griewank_functopm(position)
            return res
        
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
    
    def update_velocity(self):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                self.velocity[i][j] = self.w*self.velocity[i][j]+self.c1*self.random.random()*(self.p_best[i][j]-self.position[i][j])+self.c2*random.random()*(self.g_best[j]-self.position[i][j])  
                
    def update_position(self):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                if random.random <= self.p:
                    self.position[i][j] = self.p_best[i][j]+numpy.random.standard_cauchy([1])*abs(self.p_best[i][j]-self.l_best[i][j])
                else:
                    self.position[i][j] = self.l_best[i][j]+np.random.normal(0,1)*abs(self.p_best[i][j]-self.l_best[i][j])
                #position[i][j] = position[i][j]+velocity[i][j]
                if abs(self.position[i][j]) > 500:
                    self.position[i][j] = random.uniform(self.uppon,self.lower)
                    
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
            #update_velocity()
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
    def test_print(self):
        print self.pop_size
        print self.max_generation
        print self.group_size_set
        print self.dimension
        print self.uppon
        print self.lower
            
#test_ccpso = ccpso(par.popsize,par.max_generation,par.group_set,par.popsize,par.upper,par.lower,par.pro_pso,par.function_index)            
test_ccpso = ccpso(1000,1000,[2,5,50,100],100,500,-500,0.7,5)
test_ccpso.run_ccpso()