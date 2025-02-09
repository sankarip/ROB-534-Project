import math
from maze import Maze2D, Maze4D

maze_path='maze2.pgm'
def point_connector(point_list, orientation, velocity, stop=False):
    #0 degrees points right
    #-90 degrees points up
    #positive y is down
    #positive x is to the right
    #use distance between/angle to accelerate
    #empty list to store points in
    points=[]
    end_point_index=len(point_list)-1
    print('end point index: ', end_point_index)
    #inting to get rid of rounding errors
    for i, point in enumerate(point_list):
        print('new point index: ', i)
        if i==end_point_index:
            break
        p1=point
        p2=point_list[i+1]
        moves=0
        point_diff_x=p1[0]-p2[0]
        point_diff_y=p1[1]-p2[1]
        #while not (round(p1[0])==round(p2[0]) and round(p1[1])==round(p2[1])):
        while (abs(point_diff_x)>0.1 or abs(point_diff_y)>0.1) and moves<10:
        #for i in range(0,10):
            moves+=1
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance=(dx**2+dy**2)**0.5
            print('distance: ',distance)
            angle_rad = math.atan2(dy, dx)  # Angle in radians
            angle_deg = math.degrees(angle_rad)
            #main advantage of this vs A* is that you can have small degree changes without a bunch of neighbors
            orientation_change=angle_deg-orientation
            #ignore small changes due to rounding errors
            if math.isclose(orientation_change, 0, abs_tol=1e-9):
                orientation_change=0
            #print('orientation change: ',orientation_change)
            # loose function for desired velocity (really bad PID)
            if orientation_change!=0:
                orientation_factor=abs(orientation_change)
                #dont divide by really small numbers
                if orientation_factor<=1:
                    orientation_factor=1
                desired_velocity=distance/orientation_factor
                if desired_velocity<0.3:
                    desired_velocity=0.3
            else:
                #update to factor in current velocity
                desired_velocity=distance
                if distance<3.5 and desired_velocity>1.5:
                    desired_velocity=1.5
                if distance<2.5 and desired_velocity>1:
                    desired_velocity=1
                    print('here')
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
            print('orientation: ',orientation)
            x_change = math.cos(math.radians(orientation))*velocity
            y_change = math.sin(math.radians(orientation))*velocity
            x=p1[0]+x_change
            y=p1[1]+y_change
            p1=[x,y]
            points.append(p1)
            print('point: ',p1)
            print('velocity: ', velocity)
    return points

point_list=[[0,0], [4,7], [3,9], [1,14]] #, [1,14],[5,15]
points=point_connector(point_list, 0, 0)
m = Maze2D.from_pgm(maze_path)
m.plot_path(points, 'Maze2D')

#do we need to A* this?
#def get_neighbors(state):
    #state is x,y,v,orientation
    #assume that you either go straight or turn in 5 degree

def trig_testing():
    print(math.sin(math.radians(80)))

#trig_testing()