import numpy as np
import math
import time
import random
from maze import Maze2D  # Import Maze2D for obstacle checking
from sklearn.neighbors import NearestNeighbors
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import io

maze_path = 'maze2.pgm'


# modified to start at various states
def PSO(maze_path, start_state):
    """
    Runs Particle Swarm Optimization (PSO) to find an optimal path in the maze.

    Args:
        maze_path (str): Path to the maze file.

    Returns:
        path (list): The best optimized path snapped to the nearest grid points.
    """
    m = Maze2D.from_pgm(maze_path)
    goal_state = m.goal_state
    # start_state = m.state_from_index(m.start_index)

    solved = False
    start_time = time.time()
    attempt = 0
    # Initialize particles and velocities

    paths = []
    while not solved:
        attempt += 1
        particles, velocities = _initialize_particles(m, start_state, goal_state, n_particles=100, point_num=8)
        personal_best_positions = particles.copy()
        personal_best_fitness = np.full(100, float('inf'))
        global_best_position, global_best_fitness = None, float('inf')

        for iteration in range(10):
            fitness_values = np.array([_evaluate_fitness(m, p, obs_factor=50.0) for p in particles])

            # Update personal bests
            better_fitness_mask = fitness_values < personal_best_fitness
            personal_best_fitness[better_fitness_mask] = fitness_values[better_fitness_mask]
            personal_best_positions[better_fitness_mask] = particles[better_fitness_mask]

            # Update global best
            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < global_best_fitness:
                global_best_fitness = fitness_values[best_index]
                global_best_position = particles[best_index].copy()

            # Update velocities and particles
            velocities = _update_velocities(particles, velocities, personal_best_positions, global_best_position,
                                            iteration,
                                            max_iter=10, w_inertial=0.5, w_cognitive=2.5, w_social=1.5, max_speed=8.0)

            particles = np.clip(particles + velocities, 0, [m.cols - 1, m.rows - 1])

        path = list(global_best_position)

        # Check if the path is collision-free
        is_valid_path = all(not m.check_occupancy(tuple(point)) for point in path)
        is_valid_path &= all \
            (not m.check_hit(tuple(path[i]), np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))
        path = np.array(path)
        if is_valid_path:
            # need to update orientation in other parts of code too
            refined_path, cant_connect, orientation, velocity = point_connector(path, 90, 0, m)
            if not cant_connect:
                break

    print(f"success: {solved} on attempt {attempt}")

    path = path.tolist()
    run_time = time.time() - start_time
    print("run time:", time.time() - start_time)
    print("path distance:", np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    path_distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    return path, run_time, path_distance


### **Helper Functions** (for Initialization, Fitness, Velocity Update)
def _initialize_particles(maze, start, goal, n_particles, point_num):
    """Generates particles with random perturbations around a straight-line path."""
    start, goal = np.array(start), np.array(goal)

    # Generate a straight-line path
    direction_vector = goal - start
    unit_vector = direction_vector / np.linalg.norm(direction_vector)
    alpha = np.linspace(0, 1, point_num)
    base_positions = start + alpha[:, None] * direction_vector

    base_positions[0] = start
    base_positions[-1] = goal

    # Generate random perturbations
    particles = np.tile(base_positions, (n_particles, 1, 1))
    random_offsets = np.random.uniform(-10, 10, size=(n_particles, point_num, 2))
    particles[:, 1:-1] += random_offsets[:, 1:-1]  # Exclude start and goal

    # Clip within bounds
    particles = np.clip(particles, 0, [maze.cols - 1, maze.rows - 1])

    return particles, np.zeros_like(particles)  # Initialize zero velocities


def _evaluate_fitness(maze, particle, obs_factor):
    """Calculates the fitness of a particle based on path length and collisions."""
    total_cost = 0.0

    # Collision penalty
    occupancy = np.array([maze.check_occupancy(tuple(state)) for state in particle])
    total_cost += np.sum(occupancy) * obs_factor

    # Path length and line collision
    start_states = particle[:-1]
    end_states = particle[1:]
    deltas = end_states - start_states
    path_length = np.linalg.norm(deltas, axis=1).sum()
    hits = np.array([maze.check_hit(tuple(start), tuple(delta)) for start, delta in zip(start_states, deltas)])
    total_cost += np.sum(hits) * obs_factor

    total_cost += path_length
    return total_cost


def _update_velocities(particles, velocities, personal_best, global_best, iteration, max_iter, w_inertial, w_cognitive,
                       w_social, max_speed):
    """Updates particle velocities using the standard PSO update rule."""
    r1, r2 = np.random.rand(*velocities.shape), np.random.rand(*velocities.shape)
    w_inertial = w_inertial - (w_inertial - 0.1) * iteration / max_iter

    inertia = w_inertial * velocities
    cognitive = w_cognitive * r1 * (personal_best - particles)
    social = w_social * r2 * (global_best - particles)

    new_velocities = inertia + cognitive + social
    new_velocities[:, 0] = 0
    new_velocities[:, -1] = 0
    return np.clip(new_velocities, -max_speed, max_speed)


def predict_position(x, y, velocity, orientation, time_step=1):
    """Predicts position"""
    x_new = x + math.cos(math.radians(orientation)) * velocity * time_step
    y_new = y + math.sin(math.radians(orientation)) * velocity * time_step
    return x_new, y_new


def monitor_collisions(robot_states, goal, time_horizon=1, safety_radius=0.5):
    """
    Check collisions and return v for yielding robots
    robot_states is list of [{id, x, y, velocity, orientation}], 1 for each robot
    """
    updates = {}
    # track if something got updated
    updated = False
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
                    updated = True
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


def point_connector(point_list, orientation, velocity, maze):
    # 0 degrees points right
    # -90 degrees points up
    # positive y is down
    # positive x is to the right
    # use distance between/angle to accelerate
    # empty list to store points in
    points = []
    end_point_index = len(point_list) - 1
    # inting to get rid of rounding errors
    cant_connect = False
    velocities = []
    orientations = []
    for i, point in enumerate(point_list):
        if i == end_point_index:
            break
        if i == 0:
            p1 = point
        else:
            try:
                p1 = last_point
            except:
                print('fix this error!')
                print('point_list: ', point_list)
                print('i: ', i)
        p2 = point_list[i + 1]
        moves = 0

        point_diff_x = p1[0] - p2[0]
        point_diff_y = p1[1] - p2[1]
        # while not (round(p1[0])==round(p2[0]) and round(p1[1])==round(p2[1])):
        # changed
        while (abs(point_diff_x) > 0.15 or abs(point_diff_y) > 0.15) and moves < 15:
            # for i in range(0,10):
            moves += 1
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            # print('distance: ',distance)
            angle_rad = math.atan2(dy, dx)  # Angle in radians
            angle_deg = math.degrees(angle_rad)
            # print('desired angle: ', angle_deg)
            # main advantage of this vs A* is that you can have small degree changes without a bunch of neighbors
            orientation_change = angle_deg - orientation
            # ignore small changes due to rounding errors
            if math.isclose(orientation_change, 0, abs_tol=1e-9):
                orientation_change = 0
            # print('orientation change: ',orientation_change)
            # loose function for desired velocity (really bad PID)
            if orientation_change != 0:
                orientation_factor = abs(orientation_change)
                # dont divide by really small numbers
                if orientation_factor <= 1:
                    orientation_factor = 1
                desired_velocity = distance / orientation_factor
                if desired_velocity < 0.3:
                    desired_velocity = 0.3
            else:
                # update to factor in current velocity
                desired_velocity = distance
                if distance < 3.5 and desired_velocity > 1.5:
                    desired_velocity = 1.5
                if distance < 2.5 and desired_velocity > 1:
                    desired_velocity = 1
            velocity_change = desired_velocity - velocity
            if velocity_change > 1:
                velocity_change = 1
            if velocity_change < -1:
                velocity_change = -1
            # print(velocity_change)
            velocity += velocity_change
            velocities.append(velocity)
            if velocity > 2:
                velocity = 2
            if orientation_change > 10:
                orientation_change = 10
            if orientation_change < -10:
                orientation_change = -10
            # apply orientation change
            orientation += orientation_change
            orientations.append(orientation)
            # print('orientation: ',orientation)
            x_change = math.cos(math.radians(orientation)) * velocity
            y_change = math.sin(math.radians(orientation)) * velocity
            x = p1[0] + x_change
            y = p1[1] + y_change
            delta = (x - p1[0], y - p1[1])
            try:
                blocked = maze.check_hit(p1, delta)
            except:
                blocked = True
            if blocked:
                # print('blocked points: ', p1, (x,y))
                cant_connect = True
                break
            p1 = [x, y]
            point_diff_x = p1[0] - p2[0]
            point_diff_y = p1[1] - p2[1]
            points.append(p1)
        # used to be indented one further, should fix error
        last_point = p1
        # changed
        if abs(point_diff_x) > 0.05 or abs(point_diff_y) > 0.05:
            cant_connect = True

    return points, cant_connect, orientation, velocity  # , velocities, orientations


def point_connector_full(point_lists, orientations, velocities, collision_avoidance=True):
    # integrate collision avoidance
    # 0 degrees points right
    # -90 degrees points up
    # positive y is down
    # positive x is to the right
    # use distance between/angle to accelerate
    # empty list to store points in
    #print('FULL CONNECTOR DEBUGGING')
    final_points = [[] for _ in range(len(point_lists))]
    final_orientations = [[] for _ in range(len(point_lists))]
    end_point_indexes = []
    curr_points = []
    next_points = []
    point_indexes = []
    completed = [False] * len(point_lists)
    # print('end point index: ', end_point_index)
    # initialize starting points and points chasing
    goal_points = []
    start_time = time.time()
    first_update = True
    for j, point_list in enumerate(point_lists):
        curr_points.append(point_list[0])
        final_points[j].append(point_list[0])
        next_points.append(point_list[1])
        point_indexes.append(1)
        end_point_indexes.append(len(point_list) - 1)
        goal_points.append(point_list[end_point_indexes[j]])
    while not all(completed):
        robot_states = []
        for j, point_list in enumerate(point_lists):
            p1 = curr_points[j]
            start_point = p1
            p2 = next_points[j]
            #print('connecting: ', p1, p2)
            # point_diff_x=p1[0]-p2[0]
            # point_diff_y=p1[1]-p2[1]
            orientation = orientations[j]
            velocity = velocities[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            angle_rad = math.atan2(dy, dx)  # Angle in radians
            angle_deg = math.degrees(angle_rad)
            # main advantage of this vs A* is that you can have small degree changes without a bunch of neighbors
            orientation_change = angle_deg - orientation
            # ignore small changes due to rounding errors
            if math.isclose(orientation_change, 0, abs_tol=1e-9):
                orientation_change = 0
            # print('orientation change: ',orientation_change)
            # loose function for desired velocity (really bad PID)
            if orientation_change != 0:
                orientation_factor = abs(orientation_change)
                # dont divide by really small numbers
                if orientation_factor <= 1:
                    orientation_factor = 1
                desired_velocity = distance / orientation_factor
                if desired_velocity < 0.3:
                    desired_velocity = 0.3
            else:
                # update to factor in current velocity
                desired_velocity = distance
                if distance < 3.5 and desired_velocity > 1.5:
                    desired_velocity = 1.5
                if distance < 2.5 and desired_velocity > 1:
                    desired_velocity = 1
            velocity_change = desired_velocity - velocity
            if velocity_change > 1:
                velocity_change = 1
            if velocity_change < -1:
                velocity_change = -1
            # print(velocity_change)
            velocity += velocity_change
            if velocity > 2:
                velocity = 2
            if orientation_change > 10:
                orientation_change = 10
            if orientation_change < -10:
                orientation_change = -10
            # apply orientation change
            orientation += orientation_change
            # print('orientation: ',orientation)
            x_change = math.cos(math.radians(orientation)) * velocity
            y_change = math.sin(math.radians(orientation)) * velocity
            x = p1[0] + x_change
            y = p1[1] + y_change
            # print('moving to: ',x,y)
            p1 = [x, y]
            curr_points[j] = p1
            point_diff_x = p1[0] - p2[0]
            point_diff_y = p1[1] - p2[1]

            if completed[j]:
                velocity = 0
                orientation = 0
            else:
                velocities[j] = velocity
                orientations[j] = orientation
                final_points[j].append(p1)
                final_orientations[j].append(orientation)
            #print(final_points)
            # found point
            # or close to goal
            x_dist_to_goal = p1[0] - goal_points[j][0]
            y_dist_to_goal = p1[1] - goal_points[j][1]
            if abs(x_dist_to_goal) < 0.1 and abs(y_dist_to_goal) < 0.1:
                close_to_goal = True
            else:
                close_to_goal = False
            if close_to_goal:
                completed[j] = True
                # stop moving
                velocities[j] = 0
            #changed from or to and
            elif abs(point_diff_x) < 0.1 and abs(point_diff_y) < 0.1:
                # if at goal
                if point_indexes[j] == end_point_indexes[j]:
                    completed[j] = True
                    # stop moving
                    velocities[j] = 0
                else:
                    #print('updating')
                    next_points[j] = point_lists[j][point_indexes[j] + 1]
                    point_indexes[j] = point_indexes[j] + 1
            robot_state = {'id': j, 'x': start_point[0], 'y': start_point[1], 'velocity': velocities[j],
                           'orientation': orientations[j]}
            robot_states.append(robot_state)
        updates, updated = monitor_collisions(robot_states, (24, 24))
        if updated and collision_avoidance:
            #if first_update:
                #print('robot states leading to collision: ', robot_states)
            robots_updated = list(updates.keys())
            # if first_update:
            #     print('updating these robots: ', robots_updated)
            #     print('final_points before updating: ', final_points)
            for robot in robots_updated:
                new_velocity = updates[robot]
                # if first_update:
                #     print('robot: ', robot, ' new velocity: ', new_velocity)
                velocities[robot] = new_velocity
                old_start_point = final_points[robot][len(final_points[robot]) - 2]
                # maybe messing things up?
                old_orientation = final_orientations[robot][len(final_points[robot]) - 2]
                x_change = math.cos(math.radians(old_orientation)) * new_velocity
                y_change = math.sin(math.radians(old_orientation)) * new_velocity
                new_point = [old_start_point[0] + x_change, old_start_point[1] + y_change]
                # if first_update:
                #     print('robot: ', robot, ' new point: ', new_point)
                curr_points[robot] = new_point
                final_points[robot][len(final_points[robot]) - 1] = new_point
            #if first_update:
                #print('final_points after updating: ', final_points)
            first_update = False
    run_time = time.time() - start_time
    # sum distance of all paths
    path_distance = 0
    for path in final_points:
        for i, point in enumerate(path):
            if i > 0:
                point_before = path[i - 1]
                dist = ((point[0] - point_before[0]) ** 2 + (point[1] - point_before[1]) ** 2) ** 0.5
                path_distance += dist
    return final_points, run_time, path_distance


def get_path_from_edges(edges):
    # go through edges backwards starting at goal.
    start_point = edges[0][0]
    # get parent node and repeat until back at start
    goal_edge = edges[len(edges) - 1]
    path = []
    end_point = goal_edge[1]
    path.append(end_point)
    current_point = end_point
    while current_point[0] != start_point[0] and current_point[1] != start_point[1]:
        for edge in edges:
            if edge[1][0] == current_point[0] and edge[1][1] == current_point[1]:
                current_point = edge[0]
                path.append(current_point)
                break
    # path would be backwards
    path = list(reversed(path))
    return path


def get_rand_point(range_start, range_end, maze):
    # only get unoccupied points
    occupied = True
    while occupied:
        # 1/50 chance to sample goal
        goal_chance = random.randint(0, 49)
        if goal_chance != 25:
            # get random float and multiply across the given range
            rand_num = random.random()
            val_x = rand_num * (range_end[0] - range_start[0]) + range_start[0]
            rand_num = random.random()
            val_y = rand_num * (range_end[1] - range_start[1]) + range_start[1]
            occupied = maze.check_occupancy((val_x, val_y))
        else:
            rand_num = random.random()
            val_x = rand_num + range_end[0] - 1
            rand_num = random.random()
            val_y = rand_num + range_end[1] - 1
            occupied = maze.check_occupancy((val_x, val_y))
            # print('goal checking: ', val_x, val_y)
    return val_x, val_y


def RRT(maze_path, start_state, start_orientation, start_velocity):
    m = Maze2D.from_pgm(maze_path)
    goal_state = m.goal_state
    # start_index=m.start_index
    # start_state=m.state_from_index(start_index)
    solved = False
    vertexes = []
    vertexes.append(start_state)
    orientation_velocity = []
    # update orientation here
    orientation_velocity.append([start_orientation, start_velocity])
    edges = []
    start_time = time.time()
    while not solved:
        new_point = [get_rand_point(start_state, goal_state, m)]
        # neighbors code from scipy example
        neighbors = NearestNeighbors(n_neighbors=1)
        points_array = np.array(vertexes)
        neighbors.fit(points_array)
        distance, index = neighbors.kneighbors(new_point)
        closest_point = points_array[index[0][0]]
        current_orientation_velocity = orientation_velocity[index[0][0]]
        # print(current_orientation_velocity)
        vector_len = ((new_point[0][0] - closest_point[0]) ** 2 + (new_point[0][1] - closest_point[1]) ** 2) ** 0.5
        unit_vector = [(new_point[0][0] - closest_point[0]) / vector_len,
                       (new_point[0][1] - closest_point[1]) / vector_len]
        # multiply to spread points out
        # changed
        unit_vector = [unit_vector[0] * 2.5, unit_vector[1] * 2.5]
        new_point = [[closest_point[0] + unit_vector[0], closest_point[1] + unit_vector[1]]]
        delta = (new_point[0][0] - closest_point[0], new_point[0][1] - closest_point[1])
        try:
            blocked = m.check_hit(closest_point, delta)
        except:
            blocked = True
        # START HERE
        # put velocity and orientation in points so that you can check feasible connection
        # print(closest_point[1], new_point[0])
        points, kinodynamic_blocked, new_orientation, new_velocity = point_connector([closest_point, new_point[0]],
                                                                                     current_orientation_velocity[0],
                                                                             current_orientation_velocity[1], m)
        if not (blocked or kinodynamic_blocked):
            #all makes sense, not the issue
            #comment out when done
            # print('trying to reach: ',new_point[0])
            # print('actually reached: ', points[len(points)-1])
            # print('points between: ', points)
            # print([closest_point, new_point[0]],
            # current_orientation_velocity[0],
            # current_orientation_velocity[1])
            # print('unit vector: ',unit_vector)
            # print('new point: ',new_point)
            edges.append([closest_point, new_point[0]])
            vertexes.append(new_point[0])
            orientation_velocity.append([new_orientation, new_velocity])
            # print(vertexes)
            # only keeping the point if it can connect
            if abs(goal_state[0] - new_point[0][0]) < 1 and abs(goal_state[1] - new_point[0][1]) < 1:
                solved = True
                success = True
            current_point = new_point
        if len(vertexes) > 5000:
            solved = True
            success = False
    run_time = time.time() - start_time
    print('run time: ', run_time)
    path = get_path_from_edges(edges)
    path_distance = 0
    for i, point in enumerate(path):
        if i > 0:
            point_before = path[i - 1]
            dist = ((point[0] - point_before[0]) ** 2 + (point[1] - point_before[1]) ** 2) ** 0.5
            path_distance += dist
    print('path distance: ', path_distance)
    # m.plot_path(np.array(path), 'Maze2D')
    return path, run_time, path_distance,


def run_experiment(num_agents, num_trials, maze_path, gif=False, save=True):
    RRT_calculation_times = []
    RRT_path_distances = []
    RRT_path_times = []
    PSO_calculation_times = []
    PSO_path_distances = []
    PSO_path_times = []
    start_orientation=90
    for trial in range(num_trials):
        # store paths for collision detection for this run
        RRT_paths = []
        PSO_paths = []
        # keep track of total run time for algorithms
        RRT_time = 0
        PSO_time = 0
        # random start locations in upper left corner
        start_locations = []
        for agent in range(num_agents):
            start_location = [random.randint(0, 3), random.randint(0, 3)]
            # dont start in the same place
            while start_location in start_locations:
                start_location = [random.randint(0, 3), random.randint(0, 3)]
            start_locations.append(start_location)
            # get trial data
            RRT_path, RRT_run_time, RRT_path_distance = RRT(maze_path, start_location, start_orientation, 0)
            RRT_paths.append(RRT_path)
            RRT_time += RRT_run_time
            PSO_path, PSO_run_time, PSO_path_distance = PSO(maze_path, start_location)
            PSO_paths.append(PSO_path)
            PSO_time += PSO_run_time
        # cd for collision detections
        RRT_cd_paths, RRT_cd_time, RRT_cd_path_distance = point_connector_full(RRT_paths, [start_orientation] * num_agents,
                                                                               [0] * num_agents)
        # make gif of the first trial
        if trial == 0 and gif:
            print('making RRT animation...')
            paths_animator(RRT_cd_paths, maze_path, 'RRT_paths')
        RRT_time += RRT_cd_time
        # total calculation time
        RRT_calculation_times.append(RRT_time)
        # average path distance
        RRT_path_distances.append(RRT_cd_path_distance / num_agents)
        # average path simulation time
        total_RRT_path_time = 0
        for path in RRT_cd_paths:
            total_RRT_path_time += len(path)
        RRT_path_times.append(total_RRT_path_time / num_agents)
        PSO_cd_paths, PSO_cd_time, PSO_cd_path_distance = point_connector_full(PSO_paths, [90] * num_agents,
                                                                               [0] * num_agents)
        # make gif of the first trial
        if trial == 0 and gif:
            print('making PSO animation...')
            paths_animator(PSO_cd_paths, maze_path, 'PSO_paths')
        PSO_time += PSO_cd_time
        # total calculation time
        PSO_calculation_times.append(PSO_time)
        # average path distance
        PSO_path_distances.append(PSO_cd_path_distance / num_agents)
        # average path simulation time
        total_PSO_path_time = 0
        for path in PSO_cd_paths:
            total_PSO_path_time += len(path)
        PSO_path_times.append(total_PSO_path_time / num_agents)
    plot_with_error_bars([RRT_calculation_times, PSO_calculation_times], ['RRT Calculation Time', 'PSO Calculation Time'], y_axis_label='Seconds', title='PSO and RRT Comparison')
    plot_with_error_bars([RRT_path_distances, PSO_path_distances], ['RRT Path Length', 'PSO Path Length'], y_axis_label='Units', title='PSO and RRT Comparison')
    if save:
        save_pickle(RRT_calculation_times, 'RRT_calc_times.pkl')
        save_pickle(RRT_path_distances, 'RRT_path_dists.pkl')
        save_pickle(RRT_path_times, 'RRT_path_times.pkl')
        save_pickle(PSO_calculation_times, 'PSO_calc_times.pkl')
        save_pickle(PSO_path_distances, 'PSO_path_dists.pkl')
        save_pickle(PSO_path_times, 'PSO_path_times.pkl')
        print('data saved')


    return RRT_calculation_times, RRT_path_distances, RRT_path_times, PSO_calculation_times, PSO_path_distances, PSO_path_times


def paths_animator(paths, maze_path, output_path):
    frames = []
    m = Maze2D.from_pgm(maze_path)

    max_path_length = max(len(path) for path in paths)  # Find longest path length
    print('Frames to animate:', max_path_length)

    # Initialize agent paths to store their movement history
    agent_paths = [[] for _ in range(len(paths))]  # List of paths per agent

    for i in range(max_path_length):
        points_to_plot = []  # Stores current positions of all agents at this timestep

        for agent_id, path in enumerate(paths):
            if i < len(path):
                point = path[i]  # Get agent's position at this time step
            else:
                point = path[-1]  # If agent has finished, keep last position

            agent_paths[agent_id].append(point)  # Store path history for each agent
            points_to_plot.append(point)  # Store current position to plot

        # Pass full history of each agent to the plot function
        fig = m.plot_path_progression(agent_paths)

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')  # Save as PNG in memory
        buf.seek(0)  # Rewind the buffer to the beginning
        image = Image.open(buf)  # Open image with PIL
        frames.append(image)
        plt.close(fig)

    # Save animation as GIF
    frames[0].save(output_path + ".gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

def plot_with_error_bars(data_list, labels, y_axis_label, title):
    means = [np.mean(lst) for lst in data_list]
    print(data_list)
    std_errors = [np.std(lst, ddof=1) / np.sqrt(len(lst)) for lst in data_list]  # Mean standard error
    plt.figure(figsize=(8, 6))
    plt.bar(labels, means, yerr=std_errors, capsize=5, alpha=0.7, color='b')
    #plt.xlabel('Category')
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.show()

#save and open variables so you dont have to rerun code
def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def open_pickle(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

#run experiments
RRT_calculation_times, RRT_path_distances, RRT_path_times, PSO_calculation_times, PSO_path_distances, PSO_path_times= run_experiment(2,2, 'maze2.pgm', gif=False)

def plot_testing():
    maze_path = 'spaced_maze.pgm'
    m = Maze2D.from_pgm(maze_path)
    maze_array = np.ones((25, 25))
    maze_array[0, 0] = 0
    start = [-0.5, -0.5]
    print([int(round(start[0])), int(round(start[1]))])
    print(maze_array)
    plt.imshow(maze_array.T, cmap=plt.get_cmap('bone'), extent=[0, 25, 25, 0])
    plt.show()
    # m.plot_path([[1,1],[2,2]])

# plot_testing()

def RRT_testing():
    maze_path = 'simple_maze.pgm'
    path, _, _ = RRT(maze_path, (3, 0), 90, 0)
    path=path
    m = Maze2D.from_pgm(maze_path)
    m.plot_path(path)
    for i, point in enumerate(path):
        if i>0:
            sub_path=[path[i-1], path[i]]
    connected, _, _ = point_connector_full([path], [90], [0])
    #print(connected)
    m.plot_path(connected[0])
    points, kinodynamic_blocked, new_orientation, new_velocity = point_connector(path, 90, 0, m)
    m.plot_path(points)

#RRT_testing()

def collision_checking():
    maze_path = 'simple_maze.pgm'
    m = Maze2D.from_pgm(maze_path)
    # bad points
    point = np.float64(3.595429195484451), np.float64(5.913372577372046)
    next_point = np.float64(4.144487309269414), np.float64(7.090398711886142)
    delta = (next_point[0] - point[0], next_point[1] - point[1])
    blocked = m.check_hit(point, delta)

#collision_checking()


# RRT_calculation_times, RRT_path_distances, RRT_path_times, PSO_calculation_times, PSO_path_distances, PSO_path_times= run_experiment(5,10, 'maze2.pgm')



# rrt_calc_time=open_pickle('RRT_calc_time.pkl')
# pso_calc_time=open_pickle('PSO_calc_time.pkl')
# save_pickle(RRT_calculation_times, 'RRT_calc_time.pkl')
# save_pickle(RRT_path_distances, 'RRT_path_dist.pkl')
# save_pickle(PSO_calculation_times, 'PSO_calc_time.pkl')
# save_pickle(PSO_path_distances, 'PSO_path_dist.pkl')




# print(RRT_calculation_times, RRT_path_distances, RRT_path_times, PSO_calculation_times, PSO_path_distances, PSO_path_times)
# print(np.average(RRT_calculation_times))
# print(np.average(PSO_calculation_times))
# print(np.average(RRT_path_distances))
# print(np.average(PSO_path_distances))
# print(np.average(RRT_path_times))
# print(np.average(PSO_path_times))





