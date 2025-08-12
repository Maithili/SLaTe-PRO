from tkinter.constants import FALSE
from gpt_utils import get_gpt_response, get_gpt_response_fake, unit_test, object_masterlist, decode_movement, decode_movement_jugaad, VERBOSE, _call_gpt
import json

prompt = """
You are a helpful assistant that understands various objects used in a crafts workshop. The workshop lasts 3 hours, so make sure everything is wrapped up and returned by the end.
In a workshop with 5 tables, people can fetch objects to their table and return objects when they are done. You need to imagine what people might be doing and generate a sequence of timestamps and movements for the game. Timestamps are in the format [HH:MM] from the beginning of the workshop, and each new timestamp must be at least 10 minutes after the previous timestamp, unless the objects need to move together and are moved to the same table, such as getting paint and paintbrush together.

The following objects are available:

Base Objects:
  - coasters: wooden coaster that can be painted or etched on
  - keychain: a keychain that can be painted or etched on and decorated with flowers, gems, etc.
  - mossframe: a wooden frame that can be etched on and filled with moss and flowers, using foamballs for height
  - photoframe: a frame that can be painted or etched on and decorated with flowers, gems, etc.
  - plantpot: a pot that can be painted and decorated with moss
  - suncatcher: a suncatcher that can be painted and decorated with gems

Core Crafting (Painting):
  - cup: a cup to wash paintbrush in between uses
  - paint: a set of paints to paint the object
  - paintbrush: a paintbrush to paint the object
  - palette: a palette to use for mixing colors
  - tissuebox: a tissuebox to clean the paintbrush in between uses

Core Crafting (Etching):
  - etchpen: a burner pen that can be used to etch on the object

Core Crafting (Moss):
  - moss: a moss that can be used to fill the frame
  
Finishing Decorations:
  - flower: artificial flowers that can be used to decorate the object
  - foamball: a foamball that can be used to lift items such as flowers and moss
  - gemset: a set of rhinestone gems that can be used to decorate the object
  - glue: a glue that can be used to attach the flowers and gems to the object
  - googlyeyes: googly eyes that can be used to decorate the object
  - stones: stones that can be used to decorate the object
  - varnish: a varnish that can be used to finish wooden objects

The following tables are available:
- tableA
- tableB
- tableC
- tableD
- tableE

Here is an example of a sequence of movements:
Storyline: 
- Alice will use tableC from 00:40 to 02:30. She will paint a photoframe. 
- Bob will use tableB from 01:10 to 02:55. He will etch a keychain and decorate it with flowers and gems.

Actions:
Thought: Alice on tableC chooses a photoframe as the base object to decorate.
[00:40] tableC: Fetch photoframe
Thought: Alice on tableC chooses Painting as the core crafting activity.
[00:40] tableC: Fetch paint
Thought: Alice needs a paintbrush to paint the photoframe.
[00:40] tableC: Fetch paintbrush
Thought: Alice needs a palette to paint the photoframe.
[00:40] tableC: Fetch palette
Thought: Alice needs a cup to paint the photoframe.
[00:40] tableC: Fetch cup
Thought: Bob on tableB chooses keychain as the base object to decorate.
[01:10] tableB: Fetch keychain
Thought: Bob on tableB chooses Etching as the core crafting activity.
[01:10] tableB: Fetch etchpen
Thought: Alice is starting to paint and needs a tissuebox to clean the paintbrush in between.
[01:20] tableC: Fetch tissuebox
Thought: Alice is done painting and needs to return the tissuebox.
[02:10] tableC: Return tissuebox
Thought: Alice is done painting and needs to clean and return the palette.
[02:30] tableC: Return palette
Thought: Alice is done painting and needs to clean and return the cup.
[02:30] tableC: Return cup
Thought: Alice is done painting and needs to clean and return the paintbrush.
[02:30] tableC: Return paintbrush
Thought: Alice is done painting and needs to clean and return the paint.
[02:30] tableC: Return paint
Thought: Alice is done painting and will leave with the photoframe.
[02:30] tableC: Return photoframe
Thought: Bob is done etching and needs to clean and return the etchpen.
[02:45] tableB: Return etchpen
Thought: Bob now wants to decorate the keychain with a flower.
[02:45] tableB: Fetch flower
Thought: Bob now wants to decorate the keychain with a gemset.
[02:45] tableB: Fetch gemset
Thought: Bob needs glue to attach the gems and flowers to the keychain.
[02:45] tableB: Fetch glue
Thought: Bob is running late and needs to be done gluing the flower and gemset to the keychain and needs to clean and return the glue.
[02:55] tableB: Return glue
Thought: Bob is done decorating the keychain and needs to clean and return the flower.
[02:55] tableB: Return flower
Thought: Bob is done decorating the keychain and needs to clean and return the gemset.
[02:55] tableB: Return gemset
Thought: Bob is done decorating the keychain and will leave with the keychain.
[02:55] tableB: Return keychain
Thought: DONE

Now, generate a new sequence of movements for the workshop.
Storyline:

"""

class MovementGenerator:
    def __init__(self, objects_masterlist, table_objects=None):
        self.objects_masterlist = objects_masterlist
        self.table_objects = {"tableA": set(), "tableB": set(), "tableC": set(), "tableD": set(), "tableE": set()}
        self.movement_history = []
        self.thought_history = []
        self.prompt = prompt
        
    def make_prompt(self):
        return prompt + '\n'.join([f"{thought}\n{movement}" for thought, movement in zip(self.thought_history, self.movement_history)])
    
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
        result = get_gpt_response(self.prompt, self.objects_masterlist, self.table_objects)
        thought = result['thought']
        movement = result['action']
        open(self.jsonl_out, 'a').write(json.dumps(result) + '\n')
        if movement == 'DONE':
            return FALSE
        if movement is not None:
            parsed_movement = decode_movement_jugaad(movement)
            if parsed_movement == (None, None, None, None):
                VERBOSE and print(f"Movement {movement} failed to decode.")
                return True
            else:
                assert(f"[{parsed_movement[0]}] {parsed_movement[1]}: {parsed_movement[2]} {parsed_movement[3]}") == movement, f"Movement {movement} not decoded correctly. Got {parsed_movement}."
            self.movement_history.append(movement)
            self.thought_history.append(thought)
            self.update_state(parsed_movement)
            self.prompt += f"{thought}\n{movement}\n"
            if int(parsed_movement[0][:2]) >= 3:
                return False
        else:
            print(f"No movement generated")
            return True
        return True
    
    def sample_rollout(self, num_movements, file_out):
        result_storyline = _call_gpt(self.prompt + "\n\nLook at the above example and predict a new storyline.", stops=["Thought","[","Actions:"])
        open(file_out, "a").write(result_storyline + '\n')
        self.prompt += result_storyline + '\nActions:\n'
        self.jsonl_out = file_out.replace(".txt", ".jsonl")
        for _ in range(num_movements):
            res = self.sample_movement()
            if not res:
                break
            open(file_out, "a").write(self.thought_history[-1] + '\n' + self.movement_history[-1] + '\n')
        return self.movement_history
    
    
if __name__ == "__main__":
    # unit_test()
    movegen = MovementGenerator(object_masterlist)
    for date in range(5):
        print(f"{date} of 5")
        rollout = movegen.sample_rollout(num_movements=100, file_out=f"/coc/flash5/mpatel377/repos/SLaTe-PRO/data/GPTdataGeneration/data/{date}.txt")
        print(rollout)
        print("\n")