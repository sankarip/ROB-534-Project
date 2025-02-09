import math
from maze import Maze2D, Maze4D

maze_path='maze2.pgm'
def point_connector(p1, p2, orientation, velocity, stop=False):
    #0 degrees points right
    #-90 degrees points up
    #positive y is down
    #positive x is to the right
    #use distance between/angle to accelerate
    #empty list to store points in
    points=[]
    #inting to get rid of rounding errors
    while int(p1[0])!=int(p2[0]) and int(p1[1])!=int(p2[1]):
    #for i in range(0,10):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance=(dx**2+dy**2)**0.5
        print('distance: ',distance)
        angle_rad = math.atan2(dy, dx)  # Angle in radians
        angle_deg = math.degrees(angle_rad)
        #main advantage of this vs A* is that you can have small degree changes without a bunch of neighbors
        orientation_change=angle_deg-orientation
        print('orientation change: ',orientation_change)
        # loose function for desired velocity (really bad PID)
        if orientation_change!=0:
            orientation_factor=abs(orientation_change)
            #dont divide by really small numbers
            if orientation_factor<=1:
                orientation_factor=1
            desired_velocity=distance/orientation_factor
            if desired_velocity<0.5:
                desired_velocity=0.5
        else:
            #update to factor in current velocity
            desired_velocity=distance
            if distance<2.5 and desired_velocity>1:
                desired_velocity=1
        velocity_change=desired_velocity-velocity
        if velocity_change>1:
            velocity_change=1
        if velocity_change<-1:
            velocity_change=-1
        #print(velocity_change)
        velocity+=velocity_change
        if velocity>2:
            velocity=2
        if orientation_change>10:
            orientation_change=10
        if orientation_change<-10:
            orientation_change=-10
        #apply orientation change
        orientation+=orientation_change
        #print(orientation)
        x_change = math.cos(math.radians(orientation))*velocity
        y_change = math.sin(math.radians(orientation))*velocity
        x=p1[0]+x_change
        y=p1[1]+y_change
        p1=[x,y]
        points.append(p1)
        print('point: ',p1)
        print('velocity: ', velocity)
    return points

points=point_connector([0,0], [4,7], 0, 0)
m = Maze2D.from_pgm(maze_path)
m.plot_path(points, 'Maze2D')

#do we need to A* this?
#def get_neighbors(state):
    #state is x,y,v,orientation
    #assume that you either go straight or turn in 5 degree

def trig_testing():
    print(math.sin(math.radians(80)))

#trig_testing()