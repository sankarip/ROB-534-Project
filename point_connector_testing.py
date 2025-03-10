import math
from maze import Maze2D, Maze4D
import time
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

maze_path='maze2.pgm'

def predict_position(x, y, velocity, orientation, time_step=1):
    """Predicts position"""
    x_new = x + math.cos(math.radians(orientation)) * velocity * time_step
    y_new = y + math.sin(math.radians(orientation)) * velocity * time_step
    return x_new, y_new


def monitor_collisions(robot_states, goal, time_horizon=3, safety_radius=0.5):
    """
    Check collisions and return v for yielding robots
    robot_states is list of [{id, x, y, velocity, orientation}], 1 for each robot
    """
    updates = {}
    #track if something got updated
    updated=False
    for i, r1 in enumerate(robot_states):
        for j, r2 in enumerate(robot_states):
            if i >= j:  # Avoid duplicates
                continue

            # Compute priority  (closer to goal = higher priority)
            p1 = math.sqrt((r1["x"] - goal[0]) ** 2 + (r1["y"] - goal[1]) ** 2)
            p2 = math.sqrt((r2["x"] - goal[0]) ** 2 + (r2["y"] - goal[1]) ** 2)

            for t in range(1, time_horizon + 1):
                x1, y1 = predict_position(r1["x"], r1["y"], r1["velocity"], r1["orientation"], t)
                x2, y2 = predict_position(r2["x"], r2["y"], r2["velocity"], r2["orientation"], t)

                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if distance < safety_radius:
                    updated=True
                    # Scale velocity based on proximity
                    scale_factor = distance / safety_radius
                    new_velocity = r1["velocity"] * scale_factor

                    # Rob with lowest priorty (highest number) yileds
                    if p1 < p2:
                        updates[r2["id"]] = new_velocity
                    else:
                        updates[r1["id"]] = new_velocity
                    break

    return updates, updated

def point_connector(point_list, orientation, velocity, stop=False):
    #0 degrees points right
    #-90 degrees points up
    #positive y is down
    #positive x is to the right
    #use distance between/angle to accelerate
    #empty list to store points in
    points=[]
    end_point_index=len(point_list)-1
    #print('end point index: ', end_point_index)
    #inting to get rid of rounding errors
    cant_connect=False
    velocities=[]
    orientations=[]
    for i, point in enumerate(point_list):
        #print('new point index: ', i)
        if i==end_point_index:
            break
        if i==0:
            p1=point
        else:
            p1=last_point
        p2=point_list[i+1]
        moves=0
        point_diff_x=p1[0]-p2[0]
        point_diff_y=p1[1]-p2[1]
        #while not (round(p1[0])==round(p2[0]) and round(p1[1])==round(p2[1])):
        while (abs(point_diff_x)>0.1 or abs(point_diff_y)>0.1) and moves<20:
        #for i in range(0,10):
            moves+=1
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance=(dx**2+dy**2)**0.5
            #print('distance: ',distance)
            angle_rad = math.atan2(dy, dx)  # Angle in radians
            angle_deg = math.degrees(angle_rad)
            #print('desired angle: ', angle_deg)
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
            velocity_change=desired_velocity-velocity
            if velocity_change>1:
                velocity_change=1
            if velocity_change<-1:
                velocity_change=-1
            #print(velocity_change)
            velocity+=velocity_change
            velocities.append(velocity)
            if velocity>2:
                velocity=2
            if orientation_change>10:
                orientation_change=10
            if orientation_change<-10:
                orientation_change=-10
            #apply orientation change
            orientation+=orientation_change
            orientations.append(orientation)
            #print('orientation: ',orientation)
            x_change = math.cos(math.radians(orientation))*velocity
            y_change = math.sin(math.radians(orientation))*velocity
            x=p1[0]+x_change
            y=p1[1]+y_change
            p1=[x,y]
            point_diff_x=p1[0]-p2[0]
            point_diff_y=p1[1]-p2[1]
            points.append(p1)
            #print('point: ',p1)
            #print('velocity: ', velocity)
            last_point=p1
        if abs(point_diff_x)>0.1 or abs(point_diff_y)>0.1:
            cant_connect=True

    return points, cant_connect, orientation, velocity#, velocities, orientations


def point_connector_full(point_lists, orientations, velocities, collision_avoidance=True):
    #integrate collision avoidance
    #0 degrees points right
    #-90 degrees points up
    #positive y is down
    #positive x is to the right
    #use distance between/angle to accelerate
    #empty list to store points in
    final_points=[[] for _ in range(len(point_lists))]
    final_orientations = [[] for _ in range(len(point_lists))]
    end_point_indexes=[]
    curr_points=[]
    next_points = []
    point_indexes =[]
    completed= [False] * len(point_lists)
    #print('end point index: ', end_point_index)
    #initialize starting points and points chasing
    goal_points=[]
    for j, point_list in enumerate(point_lists):
        curr_points.append(point_list[0])
        final_points[j].append(point_list[0])
        next_points.append(point_list[1])
        point_indexes.append(1)
        end_point_indexes.append(len(point_list) - 1)
        goal_points.append(point_list[end_point_indexes[j]])
    while not all(completed):
        robot_states=[]
        for j, point_list in enumerate(point_lists):
            p1=curr_points[j]
            start_point=p1
            p2=next_points[j]
            #point_diff_x=p1[0]-p2[0]
            #point_diff_y=p1[1]-p2[1]
            orientation=orientations[j]
            velocity=velocities[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance=(dx**2+dy**2)**0.5
            #print('distance: ',distance)
            angle_rad = math.atan2(dy, dx)  # Angle in radians
            angle_deg = math.degrees(angle_rad)
            #print('desired angle: ', angle_deg)
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
            #print('orientation: ',orientation)
            x_change = math.cos(math.radians(orientation))*velocity
            y_change = math.sin(math.radians(orientation))*velocity
            x=p1[0]+x_change
            y=p1[1]+y_change
            print('moving to: ',x,y)
            p1=[x,y]
            curr_points[j]=p1
            point_diff_x=p1[0]-p2[0]
            point_diff_y=p1[1]-p2[1]


            #print('point: ',p1)
            if completed[j]:
                velocity=0
                orientation=0
            else:
                velocities[j]=velocity
                orientations[j]=orientation
                final_points[j].append(p1)
                final_orientations[j].append(orientation)
            #found point
            #or close to goal
            x_dist_to_goal=p1[0]-goal_points[j][0]
            y_dist_to_goal = p1[1] - goal_points[j][1]
            if abs(x_dist_to_goal) < 0.5 and abs(y_dist_to_goal) < 0.5:
                close_to_goal=True
            else:
                close_to_goal=False
            if close_to_goal:
                completed[j] = True
                # stop moving
                velocities[j] = 0
            elif abs(point_diff_x)<0.1 or abs(point_diff_y)<0.1:
                #if at goal
                if point_indexes[j]==end_point_indexes[j]:
                    completed[j]=True
                    #stop moving
                    velocities[j]=0
                else:
                    next_points[j]=point_lists[j][point_indexes[j]+1]
                    point_indexes[j]=point_indexes[j]+1
            robot_state={'id':j,'x':start_point[0], 'y':start_point[1], 'velocity':velocities[j],'orientation':orientations[j]}
            robot_states.append(robot_state)
        updates,updated=monitor_collisions(robot_states, (24,24))
        if updated and collision_avoidance:
            print('updating velocities')
            robots_updated=list(updates.keys())
            for robot in robots_updated:
                new_velocity=updates[robot]
                velocities[j]=new_velocity
                old_start_point=final_points[j][len(final_points[j])-2]
                #maybe messing things up?
                old_orientation=final_orientations[j][len(final_points[j])-2]
                x_change = math.cos(math.radians(old_orientation)) * new_velocity
                y_change = math.sin(math.radians(old_orientation)) * new_velocity
                new_point=[old_start_point[0]+x_change, old_start_point[1]+y_change]
                curr_points[j]=new_point
                final_points[j][len(final_points[j]) - 1]=new_point

    return final_points#, orientations, velocitys


def trig_testing():
    print(math.sin(math.radians(80)))

#trig_testing()

def get_path_from_edges(edges):
    #go through edges backwards starting at goal.
    start_point=edges[0][0]
    #get parent node and repeat until back at start
    goal_edge=edges[len(edges)-1]
    path=[]
    end_point=goal_edge[1]
    path.append(end_point)
    current_point=end_point
    while current_point[0]!=start_point[0] and current_point[1]!=start_point[1]:
        for edge in edges:
            if edge[1][0]==current_point[0] and edge[1][1]==current_point[1]:
                print(edge)
                current_point=edge[0]
                path.append(current_point)
                break
    #path would be backwards
    path=list(reversed(path))
    return path

def get_rand_point(range_start, range_end, maze):
    #only get unoccupied points
    occupied=True
    while occupied:
        #1/50 chance to sample goal
        goal_chance=random.randint(0,49)
        if goal_chance!=25:
            #get random float and multiply across the given range
            rand_num=random.random()
            val_x=rand_num*(range_end[0]-range_start[0])+range_start[0]
            rand_num=random.random()
            val_y=rand_num*(range_end[1]-range_start[1])+range_start[1]
            occupied=maze.check_occupancy((val_x, val_y))
        else:
            rand_num=random.random()
            val_x=rand_num+range_end[0]-1
            rand_num=random.random()
            val_y=rand_num+range_end[1]-1
            occupied=maze.check_occupancy((val_x, val_y))
            #print('goal checking: ', val_x, val_y)
    return val_x, val_y

def RRT(maze_path):
    m = Maze2D.from_pgm(maze_path)
    goal_state=m.goal_state
    goal_index=m.goal_index
    start_index=m.start_index
    start_state=m.state_from_index(start_index)
    solved=False
    vertexes=[]
    vertexes.append(start_state)
    orientation_velocity=[]
    orientation_velocity.append([0,0])
    edges=[]
    start_time = time.time()
    while not solved:
        new_point=[get_rand_point(start_state,goal_state, m)]
        #neighbors code from scipy example
        neighbors = NearestNeighbors(n_neighbors=1)
        points_array=np.array(vertexes)
        neighbors.fit(points_array)
        distance, index = neighbors.kneighbors(new_point)
        closest_point = points_array[index[0][0]]
        current_orientation_velocity = orientation_velocity[index[0][0]]
        vector_len=((new_point[0][0]-closest_point[0])**2+(new_point[0][1]-closest_point[1])**2)**0.5
        unit_vector=[(new_point[0][0]-closest_point[0])/vector_len,(new_point[0][1]-closest_point[1])/vector_len]
        # multiply to spread points out
        unit_vector=[unit_vector[0]*2, unit_vector[1]*2]
        new_point=[[closest_point[0]+unit_vector[0], closest_point[1]+unit_vector[1]]]

        delta=(new_point[0][0]-closest_point[0],new_point[0][1]-closest_point[1])
        try:
            blocked=m.check_hit(closest_point, delta)
        except:
            blocked=True
        #START HERE
        #put velocity and orientation in points so that you can check feasible connection
        #print(closest_point[1], new_point[0])
        points, kinodynamic_blocked, new_orientation, new_velocity = point_connector([closest_point, new_point[0]], current_orientation_velocity[0], current_orientation_velocity[1])
        if not (blocked or kinodynamic_blocked):
            #print('unit vector: ',unit_vector)
            #print('new point: ',new_point)
            edges.append([closest_point, new_point[0]])
            vertexes.append(new_point[0])
            orientation_velocity.append([new_orientation, new_velocity])
            #print(vertexes)
            #only keeping the point if it can connect
            if abs(goal_state[0]-new_point[0][0])<1 and abs(goal_state[1]-new_point[0][1])<1:
                solved=True
                success=True
            current_point=new_point
        if len(vertexes)>5000:
            solved=True
            success=False
    print('success: ', success)
    print('run time: ',  time.time()-start_time)
    print('num vertexes: ', len(vertexes))
    path=get_path_from_edges(edges)
    path_distance=0
    for i, point in enumerate(path):
        if i>0:
            point_before=path[i-1]
            dist=((point[0]-point_before[0])**2+(point[1]-point_before[1])**2)**0.5
            path_distance+=dist
    print('path distance: ', path_distance)
    #m.plot_path(np.array(path), 'Maze2D')
    return path
#point_list=[[0,0], [4,7], [1,10]]#, [1,14]] #, [1,14],[5,15]
#points=point_connector(point_list, 0, 0)
def testing3():
    m = Maze2D.from_pgm(maze_path)
    #m.plot_path(points, 'Maze2D')
    #paths=[points, [(0,0),[0,2], [2,2]]]
    #m.plot_multiple_paths(paths)
    path=RRT(maze_path)
    # filename = 'RRTpath.pkl'
    # with open(filename, 'wb') as file:
    #     pickle.dump(path, file)
    m.plot_path(path)
    points, cant_connect, orientation, velocity=point_connector(path, 0, 0)
    m.plot_path(points)
# points2=point_connector(path[0:3], 0, 0)
#m.plot_path(points2)
def testing():
    with open('RRTpath.pkl', 'rb') as file:
        path = pickle.load(file)
    path=path[1:3]
    points, cant_connect = point_connector(path, 0, 0)
    m.plot_path(points)
    print(path)
    print(points)
    print(cant_connect)

#testing()

def testing_full_path():
    #path1=[[0,0], [2,2]]
    #path2 = [[0, 0], [1, 0], [2, 0]]
    path1=RRT(maze_path)
    path2=RRT(maze_path)
    paths=[path1, path2]
    orientations=[0,0]
    velocities=[0,0]
    points=point_connector_full(paths, orientations, velocities)
    no_collsion_avoidance=point_connector_full(paths, orientations, velocities, False)
    print(points)
    m = Maze2D.from_pgm(maze_path)
    m.plot_multiple_paths(points)
    m.plot_multiple_paths(no_collsion_avoidance)

testing_full_path()






