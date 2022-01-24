# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.accum_travel_hours = 0
        self.action_space = [(1,2), (2,1),
                            (1,3), (3,1),
                            (1,4), (4,1),
                            (1,5), (5,1),
                            (2,3), (3,2),
                            (2,4), (4,2),
                            (2,5), (5,2),
                            (3,4), (4,3),
                            (3,5), (5,3),
                            (4,5), (5,4),
                            (0,0)]
        self.state_space = [[a, b, c] for a in range(1,m+1) for b in range(t) for c in range(d)]
        self.state_init = random.choice([(1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)])

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        if not state:
            return
        state_encod = [0]*(m+t+d)
        state_encod[state[0]-1]= 1
        state_encod[m+state[1]]= 1
        state_encod[m+t+state[2]]= 1
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]
        if (0, 0) not in actions:
            actions.append((0,0))
            possible_actions_index.append(20)
        return possible_actions_index,actions   

    


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #Reward = Revenue - cost
        #Revenue = fare from pickup p to drop q
        #Cost = Cost of battery from current location to pickup to drop
        
        current = state[0]
        pickup = action[0]
        drop = action[1]
        tod = state[1] #time of the day
        dow = state[2] #day of the week
        
        #this is for if in case of travelling we pass 24 hr mark 
        def new_time_day(tod,dow,travel_time):
            tod = tod + travel_time % (t - 1)
            dow = dow + (travel_time // (t - 1))
            
            if tod > (t-1):
                dow = dow + (tod // (t - 1))
                tod = tod % (t - 1)
                if dow > (d - 1):
                    dow = dow % (d - 1)  
            return tod, dow
        
        def total_travel_time(current, pickup, drop, tod, dow):
            
            if not pickup and not drop:
                return 0, 1
            
            # t1 is time required from current location to pickup
            t1 = 0
            if pickup and current != pickup:
                t1 = int(Time_matrix[current-1][pickup-1][tod][dow])

                # compute new tod and dow after travel t1
                tod, dow = new_time_day(tod, dow, t1)
            
            
            #t2 is the time of the trip from pickup to drop
            t2 = int(Time_matrix[pickup-1][drop-1][tod][dow])

            return t1, t2
       
        t1, t2 = total_travel_time(current, pickup, drop, tod, dow)
        
        
        if not pickup and not drop:
            reward = -C
        else:
            reward = R * t2 - C * (t1 + t2) 
        return reward

    

    def next_state_func(self, state, action, Time_matrix):
        
        
        current = state[0]
        pickup = action[0]
        drop = action[1]
        tod = state[1]
        dow = state[2]
        
        def total_travel_time1(current, pickup, drop, tod, dow):
            
            if not pickup and not drop:
                return 1
            
            # t1 is time required from current location to pickup
            t1 = 0
            if pickup and current != pickup:
                t1 = int(Time_matrix[current-1][pickup-1][tod][dow])

                # compute new tod and dow after travel t1
                tod, dow = new_time_day(tod, dow, t1)
            
            
            #t2 is the time of the trip from pickup to drop
            t2 = int(Time_matrix[pickup-1][drop-1][tod][dow])

            return t1+t2

        def new_time_day(tod,dow,travel_time):
            tod = tod + travel_time % (t - 1)
            dow = dow + (travel_time // (t - 1))
            
            if tod > (t-1):
                dow = dow + (tod // (t - 1))
                tod = tod % (t - 1)
                if dow > (d - 1):
                    dow = dow % (d - 1)  
            return tod, dow
        
        total_trv_time = total_travel_time1(current, pickup, drop, tod, dow)
        self.accum_travel_hours += total_trv_time
        new_tod, new_dow = new_time_day(tod, dow, total_trv_time)
        
        if not pickup and not drop:
            new_loc = state[0]
        else:
            new_loc = action[1]
        next_state = (new_loc, new_tod, new_dow)    

        return next_state 



    def reset(self):
        self.accum_travel_hours = 0
        self.state_init = random.choice([(1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)])
        return self.action_space, self.state_space, self.state_init
