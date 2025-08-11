import json
import os
import re
import random
from openai import OpenAI
client = OpenAI()

object_masterlist = [
"coasters",
"cup",
"etchpen",
"flower",
"foamball",
"gemset",
"glue",
"googlyeyes",
"keychain",
"moss",
"mossframe",
"paint",
"paintbrush",
"palette",
"photoframe",
"plantpot",
"stones",
"suncatcher",
"tissuebox",
"varnish"
]

def get_gpt_response(prompt, objects_masterlist, table_objects={}, model="gpt-5", grammar=""):
    try:
        grammar = get_grammar_for_constraint(objects_masterlist, table_objects)
        response_mssql = client.responses.create(
            input=prompt,
            model=model,
            text={"format": {"type": "text"}},
            tools=[
                {
                    "type": "custom",
                    "name": "movement_grammar",
                    "description": "Saves a movement in the format [HH:MM] tableX: Action object_name",
                    "format": {
                        "type": "grammar",
                        "syntax": "regex",
                        "definition": grammar
                    }
                },
            ],
            parallel_tool_calls=False   
        )
        try:
            result = response_mssql.output[1].content[0].text
        except AttributeError as e:
            result = response_mssql.output[1].input
        pattern = re.compile(grammar)
        match = pattern.match(result)
        if match:
            print(f"✓ Valid GPT movement: {result}")
        else:
            print(f"✗ Invalid GPT movement: {result}")
        return result
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return None

def get_gpt_response_fake(prompt, objects_masterlist, table_objects={}, model="gpt-5", grammar=""):
    return random.choice(["[00:00] tableA: Fetch coasters","[00:10] tableD: Fetch paint"])

def get_grammar_for_constraint(objects_masterlist, table_objects={}):
    """
    Write the regex grammar for object movements across tables.p
    The grammar enforces constraints directly in the regex pattern.
    
    Args:
        objects_masterlist (list): Master list of valid object names
        table_objects (dict, optional): Current state of objects on each table
        
    Returns:
        str: Regex pattern with embedded constraints
    """
    table_patterns = []
    
    for table_name in ['tableA', 'tableB', 'tableC', 'tableD', 'tableE']:
        current_objects = table_objects.get(table_name, [])
        available_objects = "|".join(current_objects)
        unavailable_objects = "|".join(set(objects_masterlist) - set(current_objects))
        
        # For each table, create patterns that enforce:
        # - Fetch: any objects NOT currently on that table
        # - Return: only objects currently on that table
        if len(available_objects) > 0:
            return_pattern = rf"\[(\d{{2}}:\d{{2}})\]\s+({table_name}):\s+(Return)\s+({available_objects})"
            table_patterns.append(return_pattern)
        if len(unavailable_objects) > 0:
            fetch_pattern = rf"\[(\d{{2}}:\d{{2}})\]\s+({table_name}):\s+(Fetch)\s+({unavailable_objects})"
            table_patterns.append(fetch_pattern)
        
    # Combine all table-specific patterns
    grammar = "|".join(table_patterns)
        
    return grammar

def get_grammar_for_decoding():
    grammar = r"\[(\d{2}:\d{2})\]\s+(table[ABCDE]):\s+(Fetch|Return)\s+([a-zA-Z]+)"
    return grammar

def decode_movement(movement):
    grammar_decoding = get_grammar_for_decoding()
    pattern = re.compile(grammar_decoding)
    match = pattern.match(movement)
    if match:
        timestamp = match.group(1)
        table = match.group(2)
        action = match.group(3)
        object_name = match.group(4)
        return timestamp, table, action, object_name
    else:
        return None, None, None, None
    
def decode_movement_jugaad(movement):
    try:
        movement = movement.replace("'", "").replace("\"", "").strip()
        timestamp, movement = movement.split("]")
        timestamp = timestamp.replace("[", "")
        table, action_object_name = movement.split(":")
        table = table.strip()
        action_object_name = action_object_name.strip()
        action = action_object_name.split(" ")[0]
        object_name = action_object_name.split(" ")[1]
        return timestamp, table, action, object_name
    except Exception as e:
        print(f"Error decoding movement: {e}")
        return None, None, None, None

def unit_test():    # Define the master list of valid objects
    objects_masterlist = object_masterlist
    import random
    table_objects = {
        "tableA": ['coasters', 'glue', 'paint', 'stones'],
        "tableB": ['coasters', 'cup', 'flower', 'palette', 'suncatcher'],
        "tableC": ['foamball', 'glue', 'varnish'],
        "tableD": ['coasters', 'etchpen', 'paint', 'photoframe', 'suncatcher'],
        "tableE": ['coasters', 'glue', 'stones'],
    }
    
    for table, objects in table_objects.items():
        print(f"  {table}: {sorted(objects)}")
    
    advanced_test_movements = [
        "tableD: Fetch paint",
        "[00:10] tableD: Fetch paint",      # Valid: paint exists on tableD
        "[00:20] tableD: Fetch nonexistent", # Invalid: object not on tableD
        "[00:30] tableC: Fetch paint",      # Valid: paint exists on tableC
        "[00:40] tableD: Return paint",     # Valid: paint can be returned to tableD
        "[00:50] tableD: Return paint",     # Invalid: paint is already on tableD
        "[00:50] tableD: Return mossframe",     # Invalid: paint is already on tableD
        "[00:50] tableD: Fetch mossframe",     # Invalid: paint is already on tableD
        "[01:00] tableF: Fetch paint",      # Invalid: tableF doesn't exist
    ]
    
    print(f"\nTesting advanced constraint movements:")
    for movement in advanced_test_movements:
        grammar_contraint = get_grammar_for_constraint(objects_masterlist, table_objects)
        pattern = re.compile(grammar_contraint)
        match_constraint = pattern.match(movement)
        if match_constraint:
            print(f"✓ Valid movement: {movement}")
        else:
            print(f"✗ Invalid movement: {movement}")
        grammar_parsing = get_grammar_for_decoding()
        pattern = re.compile(grammar_parsing)
        match = pattern.match(movement)
        if match:
            timestamp = match.group(1)
            table = match.group(2)
            action = match.group(3)
            object_name = match.group(4)
            print(f"✓ Valid movement: {movement}") 
            print(f"  Timestamp: {timestamp}")
            print(f"  Table: {table}")
            print(f"  Action: {action}")
            print(f"  Object: {object_name}")
        else:
            print(f"✗ Invalid movement: {movement}")
    
    # Show the generated regex pattern
    print(f"\nGenerated Advanced Regex Pattern:")
    advanced_grammar = get_grammar_for_constraint(objects_masterlist, table_objects)
    print(f"Pattern length: {len(advanced_grammar)} characters")
    print("Pattern preview:", advanced_grammar[:100] + "..." if len(advanced_grammar) > 100 else advanced_grammar) 
    
