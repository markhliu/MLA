# Import the turtle library
import turtle as t
import time
import numpy as np
import random
import pickle


def delivery_map():
    # Set up the screen
    try:
        t.setup(960,960,0,0) 
    except t.Terminator:        
        t.setup(960,960,0,0)
    t.hideturtle()
    t.bgcolor("white")
    t.tracer(False)
    t.title('Amazon Delivery Route')
    
    # Draw grid lines to form the streets
    t.pensize(3)
    for i in range(-450,500,100):  
        t.up()
        t.goto(i, -450)
        t.down()
        t.goto(i, 450)
        t.up()
        t.goto(-450,i)
        t.down()
        t.goto(450,i)
        t.up()
    
    # Write row and column numbers on the screen
    colrow=0
    for x in range(-450, 500, 100):
        t.goto(x+2,-479)
        t.write(colrow,font=('Arial',20,'normal'))
        t.goto(-470,x-10)
        t.write(colrow,font=('Arial',20,'normal'))  
        colrow += 1
    
    # Create a park
    t.goto(-250, -250)
    t.color('light green', 'light green')
    t.begin_fill()
    for _ in range(4):
        t.forward(400)
        t.left(90)
    t.end_fill()
    
    # Mark the Amazon Hub location
    t.goto(153, 152)
    t.color('red', 'red')
    t.write("H=(6,6)",font=('Arial',16,'normal')) 
    
    # End the graph if you click on X
    #t.done()  
    time.sleep(3)
    try:
        t.bye()
    except t.Terminator:
        print('exit turtle')
    


class Route():
    def __init__(self, start, end): 
        self.start=start
        self.end=end        
        self.done=False
        self.reward=0.0
        self.info={'prob': 1.0}
        self.showboard=False 
        self.actual_actions=[(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.action_space=[0, 1, 2, 3]
        self.observation_space=[i for i in range(91)]
        self.park=[(3, 3), (3, 4), (3, 5), 
                   (4, 3), (4, 4), (4, 5), 
                   (5, 3), (5, 4), (5, 5)]
        self.grid=[]
        for x in range(10):
            for y in range(10):
                if (x,y) not in self.park:
                    self.grid.append((x,y))
        self.state=self.grid.index(self.start)

    def reset(self):  
        self.state=self.grid.index(self.start)
        self.done=False
        self.reward=-1.0
        return self.state

    def step(self, action):
        actual_action=self.actual_actions[action]
        co_or=self.grid[self.state]
        new_co_or=(actual_action[0]+co_or[0], actual_action[1]+co_or[1])
        if actual_action[0]+co_or[0]<0 or actual_action[0]+co_or[0]>=10:
            new_co_or=co_or
        if actual_action[1]+co_or[1]<0 or actual_action[1]+co_or[1]>=10:
            new_co_or=co_or        
        if new_co_or in self.park:
            new_co_or=co_or         
        new_state=self.grid.index(new_co_or)
        if new_state==self.grid.index(self.end):
            self.reward=100.0
            self.done=True
        self.state=new_state
        return new_state, self.reward, self.done, self.info
 
    def display_board(self):
        # Set up the screen
        try:
            t.setup(960,960,0,0) 
        except t.Terminator:        
            t.setup(960,960,0,0)
        t.hideturtle()
        t.bgcolor("alice blue")
        t.tracer(False)
        t.title('Amazon Delivery Route')
        t.clear()
        t.pensize(3)
        for i in range(-450,500,100):  
            t.up()
            t.goto(i, -450)
            t.down()
            t.goto(i, 450)
            t.up()
            t.goto(-450,i)
            t.down()
            t.goto(450,i)
            t.up()
            
        # Write the row and column numbers on the screen
            colrow=0
            for x in range(-450, 500, 100):
                t.goto(x+2,-479)
                t.write(colrow,font=('Arial',20,'normal'))
                t.goto(-470,x-10)
                t.write(colrow,font=('Arial',20,'normal'))  
                colrow += 1
        
        t.goto(-250, -250)
        t.color('light green', 'light green')
        t.begin_fill()
        for _ in range(4):
            t.forward(400)
            t.left(90)
        t.end_fill()
        
        t.goto(153, 152)
        t.color('red', 'red')
        t.write("H=(6,6)",font=('Arial',16,'normal')) 
        t.color('black', 'black')
        t.up()      
        t.goto(self.grid[self.state][0]*100-450,\
               self.grid[self.state][1]*100-450)        

        t.update()            

    def render(self):
        if self.showboard==False:
            self.display_board()
            self.showboard=True   
        t.pensize(5)
        t.color('red', 'red')        

        t.down() 

        t.goto(self.grid[self.state][0]*100-450,\
               self.grid[self.state][1]*100-450)
        t.dot(20,"blue")
        t.down()
        t.update()        
        time.sleep(0.1)
        t.update()        
        t.color('black', 'black')
        t.up()
        t.pensize(1)

    def show_route(self, steps):
        if self.showboard==False:
            self.display_board()
            self.showboard=True           
        t.pensize(5)
        t.color('red', 'red')        
        t.up() 
        t.goto(self.grid[steps[0]][0]*100-450,\
               self.grid[steps[0]][1]*100-450)
        t.down() 
        for i in range(len(steps)-1):
            t.goto(self.grid[steps[i+1]][0]*100-450,\
                   self.grid[steps[i+1]][1]*100-450) 
            t.dot(20,"blue")
            t.update()        
            time.sleep(1)
        t.update()        
        t.color('black', 'black')
        t.up()
        t.pensize(1)
    
    def close(self):
        time.sleep(1)
        self.showboard==False
        try:
            t.bye()
        except t.Terminator:
            print('exit turtle')    





def gen_ps(perm,shortest_perm,shortest_route):

    blocks=[]
    for i in range(len(shortest_route)):
        blocks+=shortest_route[i]
    blocks.append((6,6))    
    
    # Set up the screen
    try:
        t.setup(960,960,0,0) 
    except t.Terminator:        
        t.setup(960,960,0,0)
    t.hideturtle()
    t.bgcolor("alice blue")
    t.tracer(False)
    t.title('Amazon Delivery Route')
    t.clear()
    t.pensize(3)
    for i in range(-450,500,100):  
        t.up()
        t.goto(i, -450)
        t.down()
        t.goto(i, 450)
        t.up()
        t.goto(-450,i)
        t.down()
        t.goto(450,i)
        t.up()
        
    # Write the row and column numbers on the screen
        colrow=0
        for x in range(-450, 500, 100):
            t.goto(x+2,-479)
            t.write(colrow,font=('Arial',20,'normal'))
            t.goto(-470,x-10)
            t.write(colrow,font=('Arial',20,'normal'))  
            colrow += 1
    
    t.goto(-250, -250)
    t.color('light green', 'light green')
    t.begin_fill()
    for _ in range(4):
        t.forward(400)
        t.left(90)
    t.end_fill()
    
    t.goto(153, 152)
    t.color('red', 'red')
    t.write("H=(6,6)",font=('Arial',16,'normal')) 
    t.color('black', 'black')
    t.up()      
    t.goto(6*100-450,\
           6*100-450)        
    
    t.update()    
    
    try:
        ts=t.getscreen() 
    except t.Terminator:
        ts=t.getscreen()
    t.hideturtle()
    
    
            
           
    t.pensize(5)
    t.color('red', 'red')        
    t.up() 
    
    
    
    for i,xy in enumerate(blocks): 
        t.down()
        t.goto(xy[0]*100-450,xy[1]*100-450) 
        t.dot(20,"blue")
        t.update()        
        time.sleep(0.1)
        ts.getcanvas().postscript(file=f"files/ch16/route{i}.ps")
        for j in range(8):
            if xy==perm[j]:
                file=f"files/ch16/s{j}.ps"
                ts.getcanvas().postscript(file=file)    
    
    



















