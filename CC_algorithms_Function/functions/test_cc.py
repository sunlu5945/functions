'''
@author: sunlu

'''

#ccpso for function named SCHWEFEL from http://www.sfu.ca/~ssurjano/schwef.html
#best fitness:-418.9829*dimension
#best x:420.9687


import numpy as np
import random
import sys
import math
import numpy.random


pop_size = 1000
max_generation = 1000
#group_size = 5
group_size_set = [2,5,10,20,50]
dimension= 100
uppon = 500
lower = -500
p = 0.7

position = []
velocity = []
p_best = []
g_best = []

group = []
l_best = []
l_best_value = []
p_best_value = []
g_best_value = sys.maxint

w = 0.7
c1 = 1.57
c2 = 1.57


random_index = random.randint(0,4)
group_size = group_size_set[random_index]

for i in range(pop_size):
    position.append([])
    velocity.append([])
    
for i in range(pop_size):
    p_best_value.append(0.0)
    for j in range(dimension):
        position[i].append(random.uniform(uppon,lower))
        velocity[i].append(random.uniform(uppon,lower))

for i in range(pop_size):
        p_best.append(position[i])
        
def cal_fit(position):
    temp_fitness_value = 0.0
    for i in range(dimension):
        temp_fitness_value = temp_fitness_value + (position[i]) * (math.sin(math.sqrt(abs(position[i]))))  
    return -temp_fitness_value
        
def init_g_best():
    for i in range(pop_size):
        p_best_value[i] =  cal_fit(p_best[i])
    g_best_value = min(p_best_value)
    best_index =  p_best_value.index(g_best_value)
    for i in range(dimension):
        g_best.append(p_best[best_index][i])

init_g_best()
# do_grouping 
def do_grouping():
    global group_size
    random_index = random.randint(0,4)
    for i in range (group_size):
        group.append([])
    # random grouping
    dimension_list = []
    for i in range (dimension):
        dimension_list.append(i)
    random.shuffle(dimension_list)
    # do grouping
    sub_dim = dimension / group_size
    remain = dimension % group_size
    if remain == 0:
        for i in range(group_size):
            for j in range(sub_dim):
                group[i].append(dimension_list[i*sub_dim+j])  
    if remain != 0:    
        for i in range(group_size):
            for j in range(sub_dim):
                group[i].append(dimension_list[i*sub_dim+j])  
        for i in range(remain):
            group[group_size-1].append(dimension_list[group_size*sub_dim-1+i])
                 
do_grouping()

def best_among_local(temp_local_best_list):
    local_best_value = min(temp_local_best_list)
    #print 'min:', local_best_value
    best_index =  p_best_value.index(local_best_value)
    return best_index

def find_local_best():
    temp_local_best_list = []
    for i in range(pop_size):
        if i == 0:
            temp_local_best_list.append(p_best_value[i+pop_size-1])
            temp_local_best_list.append(p_best_value[i])
            temp_local_best_list.append(p_best_value[i+1])
            #print 'i=0:', temp_local_best_list
        elif i+1 == pop_size:
            temp_local_best_list.append(p_best_value[i-1])
            temp_local_best_list.append(p_best_value[i])
            temp_local_best_list.append(p_best_value[(i+1)%pop_size])
            #print 'i=pop_size-1:', temp_local_best_list
            
        else:
            temp_local_best_list.append(p_best_value[i-1])
            temp_local_best_list.append(p_best_value[i])
            temp_local_best_list.append(p_best_value[i+1])
            #print 'i= others:', temp_local_best_list
        
        local_best_index = best_among_local(temp_local_best_list)
        #print '%s local_best_index:' %i, local_best_index
        l_best_value.append(p_best_value[local_best_index])
        del temp_local_best_list[:]
        for j in range(dimension):
            l_best.append(p_best[local_best_index])

find_local_best()

def cal_function_value(group_size_index,reference_particle,g_best):
    temp_position = []
    for i in range(len(g_best)):
        temp_position.append(g_best[i])
    for i in range(len(group[group_size_index])):
        dim_index = group[group_size_index][i]
        temp_position[dim_index] = reference_particle[dim_index]
    temp_function_value =  cal_fit(temp_position)
    return temp_function_value


def replace(group_size_index,ref_particle,replaced_particle):   
    temp_particle = replaced_particle 
    for i in range(len(group[group_size_index])):
        dim_index = group[group_size_index][i]
        temp_particle[dim_index] = ref_particle[dim_index]
    return temp_particle

    
def update_velocity():
    for i in range(pop_size):
        for j in range(dimension):
            velocity[i][j] = w*velocity[i][j]+c1*random.random()*(p_best[i][j]-position[i][j])+c2*random.random()*(g_best[j]-position[i][j])
            
def update_position():
    for i in range(pop_size):
        for j in range(dimension):
            if random.random <= p:
                position[i][j] = p_best[i][j]+numpy.random.standard_cauchy([1])*abs(p_best[i][j]-l_best[i][j])
            else:
                position[i][j] = l_best[i][j]+np.random.normal(0,1)*abs(p_best[i][j]-l_best[i][j])
            #position[i][j] = position[i][j]+velocity[i][j]
            if abs(position[i][j]) > 500:
                position[i][j] = random.uniform(uppon,lower)
            
for i in range (max_generation):
    current_best = g_best_value 
    for j in range(group_size):
        for k in range(pop_size):
            temp_value_1 = cal_function_value(j,position[k],g_best)
            temp_value_2 = cal_function_value(j, p_best[k],g_best)
            if temp_value_1 < temp_value_2:
                p_best_value[k] = temp_value_1
                p_best[k] = replace(j,position[k],p_best[k])
            if temp_value_2 < g_best_value:
                g_best_value = temp_value_2
                g_best = replace(j,p_best[k],g_best)
    find_local_best()
    #update_velocity()
    update_position()
    
    if (i+1) % 50== 0:
        for k in range(pop_size / 5 +1):
            for j in range(dimension):
                position[k][j] = random.uniform(uppon,lower)
    
    if(g_best_value > current_best):
        do_grouping()
    #print g_best
    print '%sth' %i, g_best_value




    







