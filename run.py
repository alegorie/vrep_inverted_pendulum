# -*- coding: utf-8 -*-

from BackProp_Python_v2 import NN
from online_trainer import OnlineTrainer
import json
import threading
from controller import * 
from time import sleep


simxFinish(-1)  # just in case, close all opened connections

# localhost:19997 connects to V-REP global remote API
addr = '127.0.0.1'
port = 19997
client = simxStart(addr, port, True, True, 5000, 5)

robot = SegwayController(client)
robot.setup("body", "leftMotor", "rightMotor")
# robot = Pioneer(rospy)
HL_size= 100  # nbre neurons of Hidden layer
network = NN(1, HL_size, 1)

choice = str(input('Do you want to load previous network? (y/n) --> '))
if choice == 'y':
    with open('last_w.json') as fp:
        json_obj = json.load(fp)

    for i in range(3):
        for j in range(HL_size):
            network.wi[i][j] = json_obj["input_weights"][i][j]
    for i in range(HL_size):
        for j in range(2):
            network.wo[i][j] = json_obj["output_weights"][i][j]

trainer = OnlineTrainer(robot, network)

choice = ''
my_counter = 0
while choice != 'y' and choice != 'n':
    choice = str(input('Do you want to learn? (y/n) --> '))

if choice == 'y':
    trainer.training = True
elif choice == 'n':
    trainer.training = False

continue_running = True
start = True
while(continue_running):
    my_counter += 1    
    print(my_counter)
    if(start):
        err = simxStartSimulation(robot.client, simx_opmode_oneshot_wait)
        start = False
        
   # thread = threading.Thread(target=trainer.train)
    trainer.running = True
    trainer.failed = False
   # thread.start()
    trainer.train()
    if(trainer.failed):
         err = simxStopSimulation(robot.client, simx_opmode_oneshot_wait)
         start = True
 
    



json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")
