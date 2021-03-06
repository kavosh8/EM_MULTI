# -*- coding: utf-8 -*-
"""
Grid-world Environment wit stochastic ghost
@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class grid_env(object):
    ''' Grid world with stochastic ghosts '''
    max_cell=5
    def __init__(self,to_plot=True,grid=False):
        world = np.zeros([self.max_cell,self.max_cell],dtype='int32')
        #world[1:6,1] = 1
        #world[1:3,4] = 1
        #world[4:6,4] = 1
        self.world = world
        self.grid = grid
        self.reset()
        self.observation_shape = np.shape(self.get_state())[0]
        
        if to_plot:
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(111,aspect='equal')
            ax1.axis('off')
            plt.xlim([-1,self.max_cell])
            plt.ylim([-1,self.max_cell])
            
            #colors = matplotlib.colors.ListerColormap()
            for i in range(self.max_cell):
                for j in range(self.max_cell):
                    if world[i,j]==1:
                        col = "black"
                    else:
                        col = "white"
                    ax1.add_patch(
                        patches.Rectangle(
                            (i,j),1,1,
                            #fill=False,
                            edgecolor='black',
                            linewidth = 2,
                            facecolor = col,),
                        )
                    if np.all([i,j] == self.ghost1):
                        self.g1 = ax1.add_artist(plt.Circle((i+0.5,j+0.5),0.3,color='red'))
                    '''
                    if np.all([i,j] == self.ghost2):
                        self.g2 = ax1.add_artist(plt.Circle((i+0.5,j+0.5),0.3,color='blue'))
                    '''                  
                    if np.all([i,j] == self.pacman):
                        self.p = ax1.add_artist(plt.Circle((i+0.5,j+0.5),0.3,color='yellow'))
            self.fig = fig
            self.ax1 = ax1
            self.fig.canvas.draw()
            
    def reset(self):
        self.pacman = np.array([0,0])
        self.ghost1 = np.array([2,2]) 
        #self.ghost2 = np.array([5,3])
        return self.get_state()
        
    def set_state(self,state):
        self.pacman = np.array(state[0:2])
        self.ghost1 = np.array(state[2:4])
        #self.ghost2 = np.array(state[4:6])
    
    def step(self,a):        
        # move pacman
        #print(self.pacman)
        self._move(self.pacman,a)
        ''' 
        # check collision
        dead = self._check_dead()
        if dead:
            r = -10
            return self.get_state(),r,dead
        ''' 
        # move ghosts
        
        wall = True
        while wall:
            a1 =  np.where(np.random.multinomial(1,[0.25,0.25,0.25,0.25]))[0] # random ghost #up down right left
            #print(self.ghost1)
            wall = self._move(self.ghost1,a1)
        
        ''' 
        # move ghosts
        wall = True
        while wall:
            a2 = np.where(np.random.multinomial(1,[0.1,0.1,0.4,0.4]))[0] # probabilistic ghost
            wall = self._move(self.ghost2,a2)
        '''
        # check collision again
        dead = self._check_dead()
        goal = np.all(self.pacman == np.array([self.max_cell-1,self.max_cell-1]))
        '''
        else:
            if np.all(self.pacman == np.array([6,6])):
                r = 10
                dead = True
                #print('Reached the goal')
            else:
                r = 0
        '''
        if goal:
            r=+10
            return self.get_state(),r,True
        elif dead:
            r = -1
            return self.get_state(),r,True
        else:
            r=-0.1
            return self.get_state(),r,False

        

    def get_state(self):
        state = np.concatenate((self.pacman,self.ghost1))
        return state
    
    def plot(self):
        self.g1.remove() 
        #self.g2.remove() 
        self.p.remove()
        
        # replot
        self.g1 = self.ax1.add_artist(plt.Circle(self.ghost1+0.5,0.3,color='red'))
        #self.g2 = self.ax1.add_artist(plt.Circle(self.ghost2+0.5,0.3,color='blue'))
        self.p = self.ax1.add_artist(plt.Circle(self.pacman +0.5,0.3,color='yellow'))
        self.fig.canvas.draw()
        
    def plot_predictions(self,world):
        for i in range(self.max_cell):
            for j in range(self.max_cell):
                for k in range(3):
                    if k==1:
                        col = "yellow"
                    elif k == 2:
                        col = "red"
                    elif k == 3:
                        col = 'blue'
                    if world[i,j,k]>0.0:
                        self.ax1.add_patch(patches.Rectangle(
                                (i,j),1,1,
                                #fill=False,
                                edgecolor='black',
                                linewidth = 2,
                                facecolor = col,
                                alpha=world[i,j,k]),
                            )
    
    def _move(self,s,a):
        s_old = np.copy(s)
        
        # move
        if int(a[0]) == 0: #up
            s[1] +=1
        elif int(a[0]) == 1: #down
            s[1] -=1
        elif int(a[0])== 2: #right
            s[0] +=1
        elif int(a[0])==3: #left
            s[0] -=1
        else: 
            raise ValueError('move not possible')
            
        # check if move is possible
        if s[0]<0 or s[0]>self.max_cell-1 or s[1]<0 or s[1]>self.max_cell-1: # out of grid
            wall = True
        elif np.all(self.world[s[0],s[1]] == 1): # wall
            wall = True
        else:
            wall = False
        
        if wall:
            # Need to repeat, put back old values
            s[0] = s_old[0]
            s[1] = s_old[1]
            return wall
        else:
            # Move to new state
            return wall
    
    def _check_dead(self):
        if np.all(self.pacman == self.ghost1): #or np.all(self.pacman == self.ghost2):
            return True
        else:
            return False
    
# Test
if __name__ == '__main__':
    grid = grid_env(True)
    s = grid.get_state()
    for i in range(10000): 
        a = random.sample(range(4),1)
        #print(a)
        s,r,dead = grid.step(a)
        #print(s)
        if not dead:
            grid.plot()
            plt.pause(0.01)
        else:
            print(r)
            s = grid.reset() 
    print(grid.get_state())
    print('Finished')
    plt.show(block=True)
