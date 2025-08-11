from gpt_utils import get_gpt_response, get_gpt_response_fake, unit_test, object_masterlist, decode_movement, decode_movement_jugaad

prompt = """
You are a helpful assistant that understands various objects used in a crafts workshop.
In a workshop with 5 tables, people can fetch objects to their table and return objects when they are done. You need to imagine what people might be doing and generate a sequence of timestamps and movements for the game. Timestamps are in the format [HH:MM] from the beginning of the workshop.
Here is an example of a sequence of movements:
[00:40] tableC: Fetch cup
[00:40] tableC: Fetch paint
[00:40] tableC: Fetch paintbrush
[00:40] tableC: Fetch palette
[00:40] tableC: Fetch photoframe
[01:10] tableB: Fetch etchpen
[01:10] tableB: Fetch keychain
[01:20] tableC: Fetch tissuebox
[02:00] tableB: Fetch coasters
[02:00] tableC: Fetch coasters
[02:00] tableC: Fetch mossframe
[02:10] tableC: Return tissuebox
[02:20] tableB: Return keychain
[02:40] tableC: Return coasters

Now, generate a new sequence of movements for the workshop.
"""

class MovementGenerator:
    def __init__(self, objects_masterlist, table_objects=None):
        self.objects_masterlist = objects_masterlist
        self.table_objects = {"tableA": set(), "tableB": set(), "tableC": set(), "tableD": set(), "tableE": set()}
        self.movement_history = []
        
    def make_prompt(self):
        return prompt + '\n'.join(self.movement_history)
    
    def update_state(self, parsed_movement):
        timestamp, table, action, object_name = parsed_movement
        try:
            assert table in self.table_objects, f"Table {table} not found in table_objects"
            assert object_name in self.objects_masterlist, f"Object {object_name} not found in objects_masterlist"
            if action == "Fetch":
                self.table_objects[table].add(object_name)
            elif action == "Return":
                self.table_objects[table].remove(object_name)
            else:
                raise ValueError(f"Invalid action: {action}")
        except AssertionError as e:
            print(f"Error updating state: {e}")
            print(f"Table: {table}, Object: {object_name}, Action: {action}")
            import pdb; pdb.set_trace()

    def sample_movement(self):
        movement = get_gpt_response(self.make_prompt(), self.objects_masterlist, self.table_objects)
        if movement is not None:
            parsed_movement = decode_movement_jugaad(movement)
            if parsed_movement == (None, None, None, None):
                print(f"Movement {movement} failed to decode.")
                return None
            else:
                assert(f"[{parsed_movement[0]}] {parsed_movement[1]}: {parsed_movement[2]} {parsed_movement[3]}") == movement, f"Movement {movement} not decoded correctly. Got {parsed_movement}."
            self.movement_history.append(movement)
            self.update_state(parsed_movement)
        else:
            print(f"No movement generated")
            return None
    
    def sample_rollout(self, num_movements=10):
        for _ in range(num_movements):
            self.sample_movement()
        return self.movement_history
    
    
if __name__ == "__main__":
    # unit_test()
    movegen = MovementGenerator(object_masterlist)
    print(movegen.sample_rollout(10))