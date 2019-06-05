# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:48:50 2019

@author: Etienne
"""

import time
import math


class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN
        self.running = False 
        self.failed = False
        self.alpha = [1]  # normalition avec limite du monde cartesien = -3m à + 3m

    def train(self):
        angle = self.robot.get_current_angle()[1] # angle[1] = angle suivant y (inclinaison du Segway)

        network_input = [0]
        network_input[0] = (angle)*self.alpha[0] 
        #Teta_t = 0

        while(self.running and self.failed==False):

            debut = time.time()
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t
            
                      
            alpha_teta = 1.0
                        
            crit_av= alpha_teta*alpha_teta*angle*angle
                       
            self.robot.set_target_velocities(command[0]*10,command[0]*10) # applique vitesses roues instant t,
            print(command)
            time.sleep(0.050) # attend delta t
            angle = self.robot.get_current_angle()[1] #  obtient nvlle pos robot instant t+1       
            network_input[0] = (angle)*self.alpha[0]

            
            crit_ap=alpha_teta*alpha_teta* angle * angle

            if self.training:
                delta_t = (time.time()-debut)

                grad = [
                   
                    -alpha_teta*alpha_teta*(angle)*delta_t,


                    -alpha_teta*alpha_teta*(angle)*delta_t
                    ]

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap <= crit_av) :
                    self.network.backPropagate(grad, 0.2,0.1) # grad, pas d'app, moment
                else :
                    #self.network.random_update(0.001)
                    self.network.backPropagate(grad, 0.2, 0.1)
                self.failed = abs(self.robot.get_current_angle()[1])>0.9
        # self.robot.set_target_velocities(0,0) # stop  apres arret  du prog d'app
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
                #Teta_t=position[2]
             
                
        
        self.running = False
