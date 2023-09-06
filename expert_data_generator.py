from environment.snake_env import SnakeEnv
import pygame
import yaml 
import heapq
import torch

def astar(start, goal, walls):
    """
    A* algorithm implementation to find the shortest path from start to goal
    on a grid with walls represented as 1s.
    """
    # Define the heuristic function as the Manhattan distance
    def heuristic(node):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    
    # Initialize the open and closed sets
    open_set = [(0, start)]
    closed_set = set()
    
    # Initialize the g score for the start node
    g_score = {start: 0}
    
    # Initialize the parent dictionary to keep track of the path
    parent = {}
    
    while open_set:
        # Get the node with the lowest f score from the open set
        current = heapq.heappop(open_set)[1]
        
        # If we've reached the goal, reconstruct the path and return it
        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        
        # Add the current node to the closed set
        closed_set.add(current)
        
        # Check the neighbors of the current node
        for neighbor in [(current[0]+1, current[1]), (current[0]-1, current[1]), (current[0], current[1]+1), (current[0], current[1]-1)]:
            # Skip neighbors that are walls or already in the closed set
            if neighbor in walls or neighbor in closed_set:
                continue
            
            # Calculate the tentative g score for the neighbor
            tentative_g_score = g_score[current] + 1
            
            # If the neighbor is not in the open set, add it and calculate its f score
            if neighbor not in [node[1] for node in open_set]:
                heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor), neighbor))
            # If the neighbor is already in the open set, update its g score if the new score is lower
            elif tentative_g_score < g_score[neighbor]:
                if((g_score[neighbor] + heuristic(neighbor), neighbor) in open_set):
                        
                    open_set.remove((g_score[neighbor] + heuristic(neighbor), neighbor))
                    heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor), neighbor))
            
            # Update the parent and g score dictionaries
            parent[neighbor] = current
            g_score[neighbor] = tentative_g_score
            
    # If we've exhausted all possible paths and haven't found the goal, return None
    return None

config = yaml.load(open("bc-irl-snake.yaml", "r"), Loader=yaml.FullLoader)
env_config = config["env"]["env_settings"]["params"]["config"]
env_config["render_mode"] = "human"
n_exp = env_config["num_exp"]

env_config["render_mode"] = "human"
snakie = SnakeEnv(env_config)
frames = []

expert_exp = {"reward": [], "action": [], "observation": [], "terminal": [], "next_observation": [], "infos": []}
snakie = SnakeEnv({**env_config})
for experience in range(n_exp):
    #run forever and take actions from keyboard and collect data about the reward

    expert_exp["observation"].append(snakie.reset())
    done = False
    action = 0 
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        walls =[(x//snakie.block_size,y//snakie.block_size) for (x,y) in snakie.snake.body[:-1]]
        current_cell = (snakie.snake.head[0]//snakie.block_size,snakie.snake.head[1]//snakie.block_size)


        path =  astar(current_cell, (snakie.apple.position[0]//snakie.block_size,snakie.apple.position[1]//snakie.block_size), walls)
        if(path is not None):
            next_cell = path[0]
        parent_cell = current_cell
        current_cell = next_cell 

        if (parent_cell[0] - current_cell[0], parent_cell[1] - current_cell[1]) == (1, 0):
            action = 2
        elif (parent_cell[0] - current_cell[0], parent_cell[1] - current_cell[1]) == (-1, 0):
            action = 3
        elif (parent_cell[0] - current_cell[0], parent_cell[1] - current_cell[1]) == (0, 1):
            action = 0
        else:
            action = 1

        probs = [-1.0]*4
        probs[action] = 0.0
        
        obs,reward,done,info= snakie.step(probs)
        probs = torch.tensor(probs)
        expert_exp["reward"].append(reward)
        expert_exp["action"].append(probs)
        expert_exp["observation"].append(obs)
        expert_exp["terminal"].append(done)
        expert_exp["next_observation"].append(obs)
        expert_exp["infos"].append({"dist_to_goal": float(len(path)) / (snakie.screen_width + snakie.screen_height), "final_obs": obs if done else None})
        if not done : 
            expert_exp["infos"][-1].pop("final_obs")
        # print(expert_exp["action"][-1], expert_exp["reward"][-1], expert_exp["terminal"][-1])
        
        snakie.render("human")
        pygame.time.wait(10)
    expert_exp["observation"] = expert_exp["observation"][:-1]
pygame.quit()

observations = torch.stack(expert_exp["observation"])

new_observation = torch.stack(expert_exp["next_observation"])

actions = torch.stack(expert_exp["action"])

dones = torch.tensor(expert_exp["terminal"])

rewards = torch.tensor(expert_exp["reward"])

tobejsoned = {"observations": observations, "actions": actions, "terminals": dones, "next_observations": new_observation , "rewards": rewards, "infos": expert_exp["infos"]}

file_path = "expert_data.pth"

torch.save(tobejsoned, file_path)
