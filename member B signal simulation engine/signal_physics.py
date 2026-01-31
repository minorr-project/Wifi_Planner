

"""
  signal mathematics for wifi optimization 
  here im gonna, find distance loss, obstacele loss, then calculate signal strength of grid and the coverage percent


"""

# CONSTANTS AND PARAMETERS

#reference signal strength at round (Dbm) generally this value is considered good for wifi signal
S0 = -30.0  

#minimum usable signal strenght (Dbm) 
S_threshold = -70

#distance attuenuation constant (dB per meter)
d_loss_k =  2.0

#general obstacle attenuation values (dB)
attenuation = {
    "FREE" : 0.0,
    "WALL" : 8.0,
    "DOOR" : 3.0,
    "WINDOW" : 2.0
}



#GEOMETRY FUNCTIONS
"""
   we take position of router and cell and then compute the straight distance between them        
"""
def distance(router, cell):
    x_r,y_r = router
    x_p,y_p = cell

    dx = x_p - x_r
    dy = y_p - y_r

    return (dx ** 2 + dy ** 2) ** 0.5


#LOSS MODELS

# 1) distance loss (Distance loss = k * distance)

def distance_loss (distance, k = d_loss_k):
    return (k * distance)

"""
   in this function we will find out the distance loss using distance loss formula
   distance_loss = k * distance

   where k is the distance attenuation constant (generally k = 2) and,
   distance is the distance between router and the cell 
"""


#2) obstacle loss ( obstacle loss = summation of all the attenuation of the obstacle 'n')

def obstacle_loss(router, cell, grid, att = attenuation):
    x_r , y_r = router
    x_p , y_p = cell 

    loss = 0.0

    #number of samples along the path
    steps = max(abs(x_p-x_r),abs(y_p-y_r))
    if steps == 0:
        return 0.0
    
    for step in range(1,steps):
        t = step/ steps
        x = int(round(x_r + t*(x_p-x_r)))
        y = int(round(y_r + t * (y_p - y_r)))


        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            cell_type = grid[y][x]
            loss += att.get(cell_type, 0.0)

    return loss    


# 3) signal strength computation (signal = s0 - distance loss - obstacle loss)

def signal_strength (router, cell, grid):
    d = distance(router, cell)
    loss_d = distance_loss(d)
    loss_o = obstacle_loss(router, cell, grid)

    signal = S0 - loss_d - loss_o

    return signal


# 4) best signal computation (computes the strongest signal recieved at a cell from multiple routers)

def best_signal(cell, routers, grid):
    signals = []

    for router in routers:
        s = signal_strength(router, cell, grid)
        signals.append(s)

    return max(signals)



# 5) coverage logic 


# to check if the cell is covered or not>? (if s > threshold , then we can say that the signal is usable)
def is_cell_covered (signal , thres = S_threshold):
    return signal >= thres


#computes coverage percentage and average signal strength
def coverage_metrics(routers, grid):
    total_free_cells = 0
    covered_cells = 0
    signal_sum = 0.0

    height = len(grid)
    width = len(grid[0])

    for y in range(height):
        for x in range(width):
            if grid[y][x] != "FREE":
                continue

            total_free_cells += 1
            cell = (x, y)

            signal = best_signal(cell, routers, grid)
            signal_sum += signal

            if is_cell_covered(signal):
                covered_cells += 1

    if total_free_cells == 0:
        return 0.0, 0.0

    coverage_percentage = (covered_cells / total_free_cells) * 100.0
    average_signal = signal_sum / total_free_cells

    return coverage_percentage, average_signal


# FITNESS INTERFACE (for genetic algorithm)

def fitness_function(routers, grid, beta = 0.0):
    
    coverage, _ = coverage_metrics(routers,grid)


    #overlap penalty can be added later if needed 
    overlap_penalty = 0.0


    fitness = coverage - beta * overlap_penalty
    return fitness




