import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing
import bind
import math
import metaurban.policy.orca_planner_utils as orca_planner_utils
from skimage import measure
import time




def generate_template_xml( mask):
    cellsize = 1
    agentdict = {"type":"orca-par",  "agent": []}

    mylist, h, w = orca_planner_utils.mask_to_2d_list(mask)
    contours = measure.find_contours(mylist, 0.5, positive_orientation='high')

    flipped_contours = []
    for contour in contours:
        contour = orca_planner_utils.find_tuning_point(contour, h)
        flipped_contours.append(contour)
    root = orca_planner_utils.write_to_xml(mylist, w, h, cellsize, flipped_contours, agentdict)
    return root

def get_speed(start_positions, positions):
    pos1 = positions[:-1]
    pos2 = positions[1:]

    pos_delta = pos2 - pos1
    speed = np.linalg.norm(pos_delta, axis=2)
    print(len(start_positions))
    print(positions.shape)
    speed = np.concatenate([np.zeros((1, len(start_positions))), speed], axis=0)
    return list(speed)
        
def set_agents(start_positions, goals, root):
    ### TODO, overwrite agent, instead of append
    ## overwrite agents' start and goal position in xml file
    agents = root.findall('./agents')[0]
    if agents.get("number") != "0":
        # print('need to overwrite agents')
        for child in agents.findall('agent'):
            agents.remove(child)
    agents.set("number", f"{len(start_positions)}")
    # num_agent = len(start_positions)

    for cnt, (pos, goal) in enumerate(zip(start_positions, goals)):
        agent = ET.Element("agent")
        agent.set('id', f'{cnt}')
        agent.set('size', f'{0.3}')
        agent.set('start.xr', f'{pos[0]}')
        agent.set('start.yr', f'{pos[1]}')
        agent.set('goal.xr', f'{goal[0]+0.5}') # magic number
        agent.set('goal.yr', f'{goal[1]+0.5}')

        agents.append(agent)
            
def run_planning(start_positions, goals, mask, num_agent, thread_id, results):
    root = generate_template_xml(mask)
    set_agents(start_positions, goals, root) 
    xml_string = ET.tostring(root, encoding='unicode')
    result = bind.demo(xml_string, num_agent)
    nexts = []
    time_length_list = []
    for v in result.values():
        nextxr = np.array(v.xr)#[:min_total_step] 
        nextyr = np.array(v.yr)#[:min_total_step]
        nextr = np.stack([nextxr, nextyr], axis=1)
        nexts.append(nextr)
        
        time_length = 0
        last_x, last_y = None, None
        flag = False
        for x, y in zip(nextxr, nextyr):
            if x == last_x and y == last_y:
                time_length_list.append(time_length)
                flag = True
                break
            else:
                last_x, last_y = x, y
                time_length += 1
        if not flag:
            time_length_list.append(time_length)
    nexts = np.stack(nexts, axis=1)
    speed = get_speed(start_positions, nexts)
    earliest_stop_pos = list(nexts)[-1]
    #print(f"Before assignment, results type: {type(results)}")
    #result = {key: convert_to_dict(value) for key, value in result.items()}
    #results[thread_id] = result
    results[thread_id] = (nexts, time_length_list, speed, earliest_stop_pos)
    #print(f"After assignment, results type: {type(results)}")


 
def get_planning(start_positions_list, masks, goals_list, num_agent_list, num_envs, roots=None):
    manager = multiprocessing.Manager()
    results = manager.list([None]*num_envs)
    processes = []
    for i in range(num_envs):
        p = multiprocessing.Process(target=run_planning, args=(start_positions_list[i], goals_list[i], masks[i], num_agent_list[i], i, results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()    
    nexts_list = []
    time_length_lists = []
    earliest_stop_pos_list = []
    speed_list = []
    for nexts, time_length_list, speed, earliest_stop_pos in results: 
        nexts_list.append(nexts)
        time_length_lists.append(time_length_list)
        speed_list.append(speed)
        earliest_stop_pos_list.append(earliest_stop_pos)

    # return next_positions, speed # --> next_positions.shape: [# steps, np.array(spawn_num,2)]
    return time_length_lists, nexts_list, speed_list, earliest_stop_pos_list




def random_start_and_end_points(engine, map_mask, num):
    st = time.time()
    ### cv2.erode
    starts = _random_points_new(map_mask, num)
    goals = _random_points_new(map_mask, num)
    #_random_points:  25: 0.1116s -0.133  | _random_points_new: 25: 0.008s

    #### visualization
    # if engine.global_config["show_mid_block_map"]:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots()
    #     plt.imshow(np.flipud(map_mask), origin='lower')   ######
    #     # plt.imshow(map_mask)
    #     fixed_goal = ax.scatter([p[0] for p in goals], [p[1] for p in goals], marker='x')
    #     fixed_start = ax.scatter([p[0] for p in starts], [p[1] for p in starts], marker='o')
    #     # plt.show()
    #     plt.savefig('./tmp.png')
    #     # import sys
    #     # sys.exit(0)
    print('random_start_and_end_points time: ', time.time()-st)
    return starts, goals



def _random_points_new(map_mask, num, min_dis=5):
        st = time.time()
        import matplotlib.pyplot as plt
        from scipy.signal import convolve2d
        import random
        from skimage import measure
        h, _ = map_mask.shape
        import metaurban.policy.orca_planner_utils as orca_planner_utils
        mylist, h, w = orca_planner_utils.mask_to_2d_list(map_mask)
        contours = measure.find_contours(mylist, 0.5, positive_orientation='high')
        flipped_contours = []
        for contour in contours:
            contour = orca_planner_utils.find_tuning_point(contour, h)
            flipped_contours.append(contour)
        int_points = []
        for p in flipped_contours:
            for m in p:
                int_points.append((int(m[1]), int(m[0])))
        def find_walkable_area(map_mask):
            # kernel = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1]], dtype=np.uint8)
            kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
            conv_result= convolve2d(map_mask/255, kernel, mode='same')
            ct_pts = np.where(conv_result==8) #8, 24
            ct_pts = list(zip(ct_pts[1], ct_pts[0]))
            # print('Len Before:', len(ct_pts))
            ct_pts = [c for c in ct_pts if c not in int_points]
            # print('Len After:', len(ct_pts))
            # plt.imshow(map_mask, cmap='gray'); plt.scatter([pt[0] for pt in ct_pts], [pt[1] for pt in ct_pts], color='red')
            # plt.grid(True); plt.show()
            return ct_pts
        selected_pts = []
        walkable_pts = find_walkable_area(map_mask)
        random.shuffle(walkable_pts)
        if len(walkable_pts) < num: raise ValueError(" Walkable points are less than spawn number! ")
        try_time = 0
        while len(selected_pts) < num:
            print(try_time)
            if try_time > 10000: raise ValueError("Try too many time to get valid humanoid points!")
            cur_pt = random.choice(walkable_pts)
            if all(math.dist(cur_pt, selected_pt) >= min_dis for selected_pt in selected_pts): 
                selected_pts.append(cur_pt)
            try_time+=1
        selected_pts = [(x[0], h - 1 - x[1]) for x in selected_pts]
        print('random_points_new time: ', time.time()-st)
        return selected_pts
