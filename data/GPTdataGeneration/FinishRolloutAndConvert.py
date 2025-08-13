import os
import json
from random import randint
import torch
import numpy as np
import matplotlib.pyplot as plt
from gpt_utils import decode_movement_jugaad

VERBOSE = False

data_dir = "/coc/flash5/mpatel377/repos/SLaTe-PRO/data/GPTdataGeneration/data"
for file in os.listdir(data_dir):
    if 'slatepro' in file:
        os.remove(os.path.join(data_dir, file))
files_list = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith(".txt")]

objects_masterlist = open("/coc/flash5/mpatel377/repos/SLaTe-PRO/object_list.txt", "r").read().splitlines()
tables = [o for o in objects_masterlist if o.startswith("table")]
supplyshelf = []
if "supplyshelf" in objects_masterlist:
    objects_masterlist.remove("supplyshelf")

surfaces = tables + supplyshelf
room = [o for o in objects_masterlist if o.startswith("room")]
objects = [o for o in objects_masterlist if o not in tables and o not in supplyshelf and o not in room]

print("Objects Masterlist:", ', '.join(objects_masterlist))
print("Tables:", ', '.join(tables))
print("Supply Shelves:", ', '.join(supplyshelf))
print("Rooms:", ', '.join(room))
print("Objects:", ', '.join(objects))
print("Surfaces:", ', '.join(surfaces))

static_graph_edges = torch.zeros(len(objects_masterlist), len(objects_masterlist), dtype=torch.float32)
active_edges_mask = torch.zeros_like(static_graph_edges, dtype=torch.float32)
for surface in surfaces:
    static_graph_edges[objects_masterlist.index(surface), objects_masterlist.index("room")] = 1.0
    for object in objects:
        active_edges_mask[objects_masterlist.index(object), objects_masterlist.index(surface)] = 1.0

dt = 10.0
print("Using dt: ", dt)

common_data = {
    'dt': dt,
    'start_time': 0.0,
    'end_time': 60*3,
    'node_classes': objects_masterlist,
    'node_categories': ['Rooms' if obj in room else 'Furniture' if obj in surfaces else 'placable_objects' for obj in objects_masterlist],
    'edge_keys': ["ON"],
    'static_nodes': [obj for obj in objects_masterlist if obj not in objects],
    'static_node_categories': ["Rooms", "Furniture"],
    'dataset_type': "RoboCraft",
    'n_stations': 5,
    'stations': ['tableA', 'tableB', 'tableC', 'tableD', 'tableE'],
    'dates': files_list,
    'active_edge_ranges': 
        [[int(np.where(active_edges_mask)[1].min()), int(np.where(active_edges_mask)[1].max())],
         [int(np.where(active_edges_mask)[0].min()), int(np.where(active_edges_mask)[0].max())]]
}

common_edge_data = {}
common_edge_data['home_graph'] = static_graph_edges.clone()
for obj in objects:
    common_edge_data['home_graph'][objects_masterlist.index(obj), objects_masterlist.index('room')] = 1.0
common_edge_data['nonstatic_edges'] = active_edges_mask.clone()
common_edge_data['seen_edges'] = torch.zeros_like(active_edges_mask, dtype=torch.float32)

common_embedding_map = {}


def time_hhmm_to_seconds(time_hhmm):
    time_hhmm = str(time_hhmm)
    return int(time_hhmm[:2]) * 3600 + int(time_hhmm[-2:]) * 60

def time_seconds_to_hhmm(time_seconds, return_human_readable=False):
    time_seconds = int(time_seconds)
    if return_human_readable:
        return f"{time_seconds//3600:02d}:{(time_seconds%3600)//60:02d}"
    else:
        return f"{time_seconds//3600:02d}{(time_seconds%3600)//60:02d}"



class FinisherAndConverter:
    def __init__(self, filename):
        self.filename = filename
        self.filename_out = filename.split(".")[0] + "_slatepro"
        self.objects_masterlist = objects_masterlist
        self.table_objects = {"tableA": set(), "tableB": set(), "tableC": set(), "tableD": set(), "tableE": set()}
        self.sg_current = torch.zeros(len(objects_masterlist), len(objects_masterlist), dtype=torch.float32)
        self.occupancy_current = torch.zeros((5), dtype=torch.float32)
        self.movements_parsed = []
        self.movement_history = []
        self.thought_current = ""
        self.person_current = ""
        self.sg_list = [self.sg_current]
        self.occupancy_list = [self.occupancy_current]
        self.person_times = {}
        self.person_things = {}
        
    def update_state(self, parsed_movement):
        timestamp, table, action, object_name = parsed_movement
        try:
            assert table in self.table_objects, f"Table {table} not found in table_objects"
            assert object_name in self.objects_masterlist, f"Object {object_name} not found in objects_masterlist"
            if action == "Fetch":
                self.table_objects[table].add(object_name)
                if self.person_current not in self.person_things:
                    self.person_things[self.person_current] = []
                self.person_things[self.person_current].append((table, object_name))
            elif action == "Return":
                self.table_objects[table].remove(object_name)
                if (table, object_name) in self.person_things[self.person_current]:
                    self.person_things[self.person_current].remove((table, object_name))
                else:
                    for person in self.person_things.keys():
                        if (table, object_name) in self.person_things[person]:
                            self.person_things[person].remove((table, object_name))
                            self.person_current = person
                            break
            else:
                raise ValueError(f"Invalid action: {action}")
        except AssertionError as e:
            print(f"Error updating state: {e}")
            print(f"Table: {table}, Object: {object_name}, Action: {action}")

    def read_rollout(self):
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith("- "):
                    line = line.replace("- ", "")
                    person_name = line.split("will use")[0].strip()
                    line = line.split("will use")[1].strip()
                    table_name = line.split("from")[0].strip()
                    line = line.split("from")[1].strip()
                    start_time = time_hhmm_to_seconds(line.split("to")[0].strip())
                    line = line.split("to")[1].strip()
                    end_time = time_hhmm_to_seconds(line.split(".")[0].strip())
                    self.person_times[person_name] = (table_name, start_time, end_time, randint(1,5))
                if line.startswith("Thought"):
                    self.thought_current = line.split("Thought: ")[1].strip()
                    self.person_current = [p for p in self.person_times.keys() if p in self.thought_current]
                    if len(self.person_current) >= 1:
                        self.person_current = self.person_current[0]
                    else:
                        import pdb; pdb.set_trace()
                if line.startswith("["):
                    parsed_movement = decode_movement_jugaad(line)
                    self.update_state(parsed_movement)
                    self.movements_parsed.append({
                        "timestamp": time_hhmm_to_seconds(parsed_movement[0]),
                        "table": parsed_movement[1],
                        "action": parsed_movement[2],
                        "object": parsed_movement[3],
                        "person": self.person_current,
                    })
                else:
                    # print(f"Ignoring line: {line}")
                    continue
        VERBOSE and print(f"Remaining person_times: {self.person_times}")
        VERBOSE and print(f"Remaining person_things: {self.person_things}")
        for person in self.person_times.keys():
            if person not in self.person_things:
                continue
            things_to_return = self.person_things[person].copy()
            for obj in things_to_return:
                parsed_movement = (self.person_times[person][2], obj[0], "Return", obj[1])
                self.movements_parsed.append({
                        "timestamp": parsed_movement[0],
                        "table": parsed_movement[1],
                        "action": parsed_movement[2],
                        "object": parsed_movement[3],
                        "person": person,
                    })
                self.update_state(parsed_movement)
                VERBOSE and print(f"Action: {parsed_movement}")
        self.movements_parsed.sort(key=lambda x: x['timestamp'])
        
    def generate_slatepro_data(self):
        nodes = torch.tensor([i for i in range(len(objects_masterlist))], dtype=torch.float32)
        times = torch.arange(common_data['start_time']-common_data['dt'], common_data['end_time']+1, common_data['dt'], dtype=torch.float32)
        scene_graph_list = [torch.zeros(len(objects_masterlist), len(objects_masterlist), dtype=torch.float32)]
        activity_list = [torch.zeros((5), dtype=torch.float32)]
        fig, ax = plt.subplots(1, len(times))
        ax = ax.flatten()
        number_of_movements = 0
        number_of_possible_movements = 0
        average_num_movements = []
        average_prob_of_movement = []
        ax[0].set_yticks(np.arange(0, len(objects_masterlist)), labels=objects_masterlist, fontsize=10)
        movements = ""
        movement_idx = 0
        fig, ax = plt.subplots(1, len(times))
        fig_diff, ax_diff = plt.subplots(1, len(times))
        ax = ax.flatten()
        ax_diff = ax_diff.flatten()
        ax[0].set_yticks(np.arange(0, len(objects_masterlist)), labels=objects_masterlist, fontsize=10)
        for idx, timestamp_slatepro in enumerate(times):
            new_sg = scene_graph_list[-1].clone()
            new_occupancy = activity_list[-1].clone()
            while movement_idx < len(self.movements_parsed) and self.movements_parsed[movement_idx]['timestamp'] <= timestamp_slatepro*60:
                object_name = self.movements_parsed[movement_idx]['object']
                table_name = self.movements_parsed[movement_idx]['table']
                change = -1 if self.movements_parsed[movement_idx]['action'] == "Return" else 1
                new_sg[objects_masterlist.index(object_name), objects_masterlist.index(table_name)] += change
                movement_idx += 1
            for person in self.person_times.keys():
                if self.person_times[person][1] < timestamp_slatepro*60 and self.person_times[person][2] >= timestamp_slatepro*60:
                    new_occupancy[tables.index(self.person_times[person][0])] += self.person_times[person][3]
            sg_viz = new_sg.clone()-scene_graph_list[-1]
            number_of_movements += sg_viz.sum().abs()
            number_of_possible_movements += torch.numel(sg_viz[active_edges_mask.to(bool)])
            if torch.any(sg_viz > 0):
                xx, yy = np.where(sg_viz > 0)
                for obj_x, obj_y in zip(xx, yy):
                    movements += f"[{(time_seconds_to_hhmm(timestamp_slatepro*60, return_human_readable=True))}] {objects_masterlist[obj_y]}: Fetch {objects_masterlist[obj_x]}\n"
            if torch.any(sg_viz < 0):
                xx, yy = np.where(sg_viz < 0)
                for obj_x, obj_y in zip(xx, yy):
                    movements += f"[{(time_seconds_to_hhmm(timestamp_slatepro*60, return_human_readable=True))}] {objects_masterlist[obj_y]}: Return {objects_masterlist[obj_x]}\n"
            print(movements)
            sg_viz[sg_viz > 0] = 1
            sg_viz[0,0] = 1
            ax_diff[idx].imshow(sg_viz, cmap='viridis', vmin=-1, vmax=1)
            ax[idx].imshow(new_sg, cmap='viridis', vmin=0, vmax=1)
            # ax[idx].imshow(active_edges_mask, cmap='gray', alpha=0.3)
            ax_diff[idx].set_title(f"{int(timestamp_slatepro)} mins")
            ax[idx].set_title(f"{int(timestamp_slatepro)} mins")
            ax[idx].set_xticks(np.arange(0, len(objects_masterlist)), labels=objects_masterlist, fontsize=10, rotation=90)
            ax[idx].set_xlim(np.where(active_edges_mask)[1].min(), np.where(active_edges_mask)[1].max())
            ax[idx].set_ylim(np.where(active_edges_mask)[0].min(), np.where(active_edges_mask)[0].max())
            ax_diff[idx].set_xticks(np.arange(0, len(objects_masterlist)), labels=objects_masterlist, fontsize=10, rotation=90)
            ax_diff[idx].set_xlim(np.where(active_edges_mask)[1].min(), np.where(active_edges_mask)[1].max())
            ax_diff[idx].set_ylim(np.where(active_edges_mask)[0].min(), np.where(active_edges_mask)[0].max())
            scene_graph_list.append(new_sg)
            activity_list.append(new_occupancy)
        
        if movement_idx < len(self.movements_parsed):
            VERBOSE and print(f"Warning: {len(self.movements_parsed)-movement_idx} movements not parsed")
            import pdb; pdb.set_trace()
        
        average_num_movements.append(number_of_movements)
        average_prob_of_movement.append(number_of_movements/number_of_possible_movements)
        print(f"Number of movements recorded: {number_of_movements}")
        print(f"Probability of object movement: {number_of_movements/number_of_possible_movements} ({number_of_movements}/{number_of_possible_movements})")

        fig.set_size_inches(1.5*len(times), 5)
        fig.tight_layout()
        fig_diff.set_size_inches(1.5*len(times), 5)
        fig_diff.tight_layout()
        fig.savefig(self.filename_out+".png")
        fig_diff.savefig(self.filename_out+"_diff.png")
        plt.close()
        
        times = torch.cat([times[:1]-common_data['dt'], times])
        # times = torch.cat([times[:1]-common_data['dt'], times, times[-1:]+common_data['dt']])
        scene_graph_list = [scene_graph_list[0]]+scene_graph_list
        activity_list = [activity_list[0]]+activity_list
        
        torch.save({
                'nodes': nodes,
                'edges': torch.stack(scene_graph_list, dim=0),
                'times': times,
                'active_edges': active_edges_mask,
                'activity': torch.stack(activity_list, dim=0),
            }, self.filename_out+".pt")
        open(self.filename_out+".txt", "w").write(movements)
        
        print(f"Written to {self.filename_out+'.pt'}")
        
    def run(self):
        self.read_rollout()
        self.generate_slatepro_data()


if __name__ == "__main__":
    for file in files_list:
        finisher_and_converter = FinisherAndConverter(os.path.join(data_dir, file+".txt"))
        finisher_and_converter.run()
    os.system("cp /coc/flash5/mpatel377/repos/SLaTe-PRO/data/GPTdataGeneration/data/*.pt /coc/flash5/mpatel377/repos/SLaTe-PRO/data/RoboCraft/processed_seqLM_coarse/train")
