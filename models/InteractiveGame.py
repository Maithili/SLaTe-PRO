import os
import sys
import json
import torch
import numpy as np
from adict import adict
import re
from datetime import datetime, timedelta
import argparse
import yaml

# Add paths for imports
sys.path.append('helpers')
sys.path.append('models')

from MultiModalUserTracking import MultiModalUserTrackingModule
from encoders import TimeEncodingOptions
from loader_sequential import OneHotEmbedder, IdentityEmbedder, CustomObjectEmbedder, OneHotEmbedder_tableInvariant

VERBOSE = os.environ.get('$MAITHILIS_CRAZY_ENV', 'not crazy').lower() == 'crazy'

class InteractiveGame:
    def __init__(self, model_weights_path, config_path=None):
        """
        Initialize the interactive game with a trained model.
        
        Args:
            model_weights_path (str): Path to the model weights file (.pt)
            config_path (str): Path to the configuration file (.json)
        """
        self.model_weights_path = model_weights_path
        self.config_path = config_path
        
        # Initialize game state
        self.current_time = 0  # Start at 0 minutes
        self.dt = 10  # 10-minute intervals
        self.max_time = 180  # 3 hours (180 minutes)
        
        # Object and table setup (based on FinishRolloutAndConvert.py)
        self.objects_masterlist = open("/coc/flash5/mpatel377/repos/SLaTe-PRO/object_list.txt").read().splitlines()
        
        self.tables = ["tableA", "tableB", "tableC", "tableD", "tableE"]
        self.surfaces = self.tables
        self.room = ["room"]
        
        # Initialize scene graph state
        self.scene_graph = torch.zeros(len(self.objects_masterlist), len(self.objects_masterlist), dtype=torch.float32)
        self.active_edges_mask = torch.zeros_like(self.scene_graph, dtype=torch.float32)
        
        # Set up static relationships
        for surface in self.surfaces:
            self.scene_graph[self.objects_masterlist.index(surface), self.objects_masterlist.index("room")] = 1.0
            for obj in self.objects_masterlist:
                if obj not in self.surfaces and obj not in self.room:
                    self.active_edges_mask[self.objects_masterlist.index(obj), self.objects_masterlist.index(surface)] = 1.0
        
        # Initialize object locations (all objects start in room)
        for obj in self.objects_masterlist:
            if obj not in self.surfaces and obj not in self.room:
                self.scene_graph[self.objects_masterlist.index(obj), self.objects_masterlist.index("room")] = 1.0
        
        # Initialize activity state (5 tables)
        self.activity_state = torch.zeros(5, dtype=torch.float32)
        
        # Store entire history of scene graphs, activities, and times
        self.scene_graph_history = [self.scene_graph.clone()]
        self.activity_history = [self.activity_state.clone()]
        self.time_history = [0]  # Start at 0 minutes
        
        # Time encoding
        self.time_options = TimeEncodingOptions()
        self.time_encoder = self.time_options('sine_informed')
        
        # Load model
        self.model = None
        with open(self.config_path, 'r') as f:
            print(f"Loading config from {self.config_path}")
            if '.yaml' in self.config_path:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                self.config = json.load(f)
            self.config = adict(self.config)
        self.load_model()
        
        # Node and activity embedders
        if self.config.node_embedder == 'one_hot':
            self.node_embedder = OneHotEmbedder(self.objects_masterlist)
        elif self.config.node_embedder == 'one_hot_table_invariant':
            self.node_embedder = OneHotEmbedder_tableInvariant(self.objects_masterlist)
        elif self.config.node_embedder == 'custom_object_concat':
            self.node_embedder = CustomObjectEmbedder(self.objects_masterlist, aggregation='concat')
        elif self.config.node_embedder == 'custom_object_sum':
            self.node_embedder = CustomObjectEmbedder(self.objects_masterlist, aggregation='sum')
        else:
            raise ValueError(f"Invalid node embedder: {self.params['node_embedder']}")
        self.activity_embedder = IdentityEmbedder()
        
        
        # Game history
        self.action_history = []
        
    def load_model(self):
        """Load the trained model and configuration."""
        try:
            # Create model instance
            self.model = MultiModalUserTrackingModule(model_configs=self.config, original_model=False)
            
            # Load weights
            if torch.cuda.is_available():
                weights = torch.load(self.model_weights_path, map_location='cuda')
                self.model = self.model.cuda()
            else:
                weights = torch.load(self.model_weights_path, map_location='cpu')
            
            self.model.load_state_dict(weights)
            self.model.eval()
            
            VERBOSE and print(f"✓ Model loaded successfully from {self.model_weights_path}")
            VERBOSE and print(f"✓ Configuration loaded from {self.config_path}")
            
        except Exception as e:
            VERBOSE and print(f"✗ Error loading model: {e}")
            raise
    
    def get_default_config(self):
        """Get default configuration if none is provided."""
        return {
            'c_len': 64,
            'n_len': len(self.objects_masterlist),
            'n_nodes': len(self.objects_masterlist),
            'n_stations': 5,
            'latent_predictor_type': 'lstm',
            'lookahead_steps': 3,
            'movement_inertia': 0.1,
            'addtnl_time_context': False,
            'learn_latent_magnitude': False
        }
    
    def decode_movement(self, movement_input):
        """
        Decode user input in the format [HH:MM] tableX: Action object_name
        
        Args:
            movement_input (str): User input string
            
        Returns:
            tuple: (timestamp, table, action, object_name) or (None, None, None, None) if invalid
        """
        try:
            # Remove quotes and clean up
            movement_input = movement_input.replace("'", "").replace("\"", "").strip()
            
            # Parse timestamp
            if not movement_input.startswith("["):
                return None, None, None, None
            
            timestamp_part, rest = movement_input.split("]", 1)
            timestamp = timestamp_part.replace("[", "")
            
            # Parse table and action
            if ":" not in rest:
                return None, None, None, None
                
            table_part, action_object = rest.split(":", 1)
            table = table_part.strip()
            action_object = action_object.strip()
            
            # Parse action and object
            parts = action_object.split()
            if len(parts) < 2:
                return None, None, None, None
                
            action = parts[0].strip()
            object_name = parts[1].strip()
            
            # Validate components
            if table not in self.tables:
                VERBOSE and print(f"✗ Invalid table: {table}. Valid tables: {', '.join(self.tables)}")
                return None, None, None, None
                
            if action not in ["Fetch", "Return"]:
                VERBOSE and print(f"✗ Invalid action: {action}. Valid actions: Fetch, Return")
                return None, None, None, None
                
            if object_name not in self.objects_masterlist:
                VERBOSE and print(f"✗ Invalid object: {object_name}. Valid objects: {', '.join(self.objects_masterlist)}")
                return None, None, None, None
            
            return timestamp, table, action, object_name
            
        except Exception as e:
            VERBOSE and print(f"✗ Error parsing movement: {e}")
            return None, None, None, None
    
    def time_to_minutes(self, time_str):
        """Convert HH:MM format to minutes since start."""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return 0
    
    def minutes_to_time(self, minutes):
        """Convert minutes since start to HH:MM format."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def update_state(self, timestamp, table, action, object_name):
        """
        Update the scene graph state based on the action.
        
        Args:
            timestamp (str): Time of action
            table (str): Table name
            action (str): Fetch or Return
            object_name (str): Object name
        """
        try:
            # Convert timestamp to minutes
            action_time = self.time_to_minutes(timestamp)
            
            # Validate action
            obj_idx = self.objects_masterlist.index(object_name)
            table_idx = self.objects_masterlist.index(table)
            
            if action == "Fetch":
                # Check if object can be fetched (not already on table)
                if self.scene_graph[obj_idx, table_idx] == 1:
                    VERBOSE and print(f"✗ Cannot fetch {object_name} - it's already on {table}")
                    return False
                
                # Add to table
                self.scene_graph[obj_idx, table_idx] = 1
                
                VERBOSE and print(f"✓ Fetched {object_name} to {table}")
                
            elif action == "Return":
                # Check if object is on table
                if self.scene_graph[obj_idx, table_idx] == 0:
                    VERBOSE and print(f"✗ Cannot return {object_name} - it's not on {table}")
                    return False
                
                # Remove from table
                self.scene_graph[obj_idx, table_idx] = 0
                
                VERBOSE and print(f"✓ Returned {object_name} from {table}")
            
            # Record action
            self.action_history.append({
                'timestamp': timestamp,
                'time_minutes': action_time,
                'table': table,
                'action': action,
                'object': object_name
            })
            
            # Add current state to history
            self.scene_graph_history.append(self.scene_graph.clone())
            self.activity_history.append(self.activity_state.clone())
            self.time_history.append(action_time)
            
            return True
            
        except Exception as e:
            VERBOSE and print(f"✗ Error updating state: {e}")
            return False
    
    def get_model_prediction(self, num_steps=1):
        """
        Get model prediction for the next few time steps.
        
        Args:
            num_steps (int): Number of steps to predict ahead
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            return None
            
        # try:
        self.model.reset_validation()
        with torch.no_grad():
            # Prepare input data with full history
            batch = self.prepare_batch_for_model()
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Run inference
            self.model.eval()
            predictions = self.model.evaluate_prediction(batch, num_steps=num_steps)
        
        return predictions[:,-1:,:,:]
                
            # except Exception as e:
            #     VERBOSE and print(f"✗ Error getting model prediction: {e}")
            #     return None
    
    def prepare_batch_for_model(self):
        """Prepare current state and history as a batch for the model."""
        # Create a single batch with the full history
        batch_size = 1
        sequence_length = len(self.scene_graph_history)
        
        # Ensure we have at least 2 timesteps for the model
        if sequence_length < 2:
            # Pad with initial state if needed
            while len(self.scene_graph_history) < 2:
                self.scene_graph_history.insert(0, self.scene_graph_history[0].clone())
                self.activity_history.insert(0, self.activity_history[0].clone())
                self.time_history.insert(0, 0)
            sequence_length = len(self.scene_graph_history)
        
        # Prepare edges (scene graph history) - batch x sequence x nodes x nodes
        edges = torch.stack(self.scene_graph_history, dim=0).unsqueeze(0)
        
        # Prepare node features - batch x sequence x nodes x features
        nodes = torch.arange(len(self.objects_masterlist), dtype=torch.float32)
        node_features = self.node_embedder(nodes).unsqueeze(0).repeat(sequence_length, 1, 1)
        node_features = node_features.unsqueeze(0)  # Add batch dimension
        
        # Prepare activity features - batch x sequence x activities
        activity_features = torch.stack(self.activity_history, dim=0).unsqueeze(0)
        
        # Prepare time features - batch x sequence x time_features
        time_features = self.time_encoder(torch.tensor(self.time_history)).unsqueeze(0)
        
        # Prepare dynamic edges mask - batch x sequence x nodes x nodes
        dynamic_edges_mask = self.active_edges_mask.unsqueeze(0).repeat(sequence_length, 1, 1).unsqueeze(0)
        
        # Prepare activity mask - batch x sequence x activities
        activity_mask = torch.ones_like(activity_features).to(bool)
        
        batch = {
            'edges': edges,
            'node_features': node_features,
            'node_ids': nodes.unsqueeze(0).repeat(sequence_length, 1, 1).unsqueeze(0),
            'activity_features': activity_features,
            'activity_ids': torch.stack(self.activity_history, dim=0).unsqueeze(0),
            'time_features': time_features,
            'time': torch.tensor(self.time_history).unsqueeze(0),
            'dynamic_edges_mask': dynamic_edges_mask,
            'activity_mask_drop': activity_mask,
            'node_embedder': self.node_embedder,
            'activity_embedder': self.activity_embedder,
        }
        
        return batch
    
    def display_current_state(self):
        """Display the current state of the game."""
        print(f"\n{'='*60}")
        print(f"Current Time: {self.minutes_to_time(self.current_time)}")
        print(f"Time Step: {self.current_time // self.dt + 1}/{(self.max_time // self.dt) + 1}")
        print(f"History Length: {len(self.scene_graph_history)} timesteps")
        print(f"{'='*60}")
        
        # Display object locations
        print("\nObjects Available:")
        ## Print object names in 3 columns
        for i in range(0, len(self.objects_masterlist), 6):
            if len(self.objects_masterlist) > i + 6:
                print(f"  {self.objects_masterlist[i]:15}  {self.objects_masterlist[i+1]:15}  {self.objects_masterlist[i+2]:15}  {self.objects_masterlist[i+3]:15}  {self.objects_masterlist[i+4]:15}  {self.objects_masterlist[i+5]:15}")
            else:
                for j in range(i, len(self.objects_masterlist)):
                    print(f"  {self.objects_masterlist[j]:15}", end="")
                print()
        
        # Display table contents
        print("\nTable Contents:")
        for table in self.tables:
            table_idx = self.objects_masterlist.index(table)
            objects_on_table = []
            for obj in self.objects_masterlist:
                if obj not in self.surfaces and obj not in self.room:
                    if self.scene_graph[self.objects_masterlist.index(obj), table_idx] == 1:
                        objects_on_table.append(obj)
            
            if objects_on_table:
                print(f"  {table:8}: {', '.join(objects_on_table)}")
            else:
                print(f"  {table:8}: (empty)")
        
        # Display recent actions
        if self.action_history:
            print(f"\nRecent Actions:")
            for action in self.action_history[-5:]:  # Show last 5 actions
                print(f"  [{action['timestamp']}] {action['table']}: {action['action']} {action['object']}")
    
    def get_suggestions(self):
        """Get model predictions as suggestions for the user."""
        ## If scene graph history has fewer than 3 timesteps, return empty list
        if len(self.scene_graph_history) <= 3:
            return []
        
        predictions = self.get_model_prediction(num_steps=1)
        
        ## Change threshold 
        predictions[predictions.abs() < 0.5] = 0
        suggestions = []
        print("AI Suggestions:")
        if (predictions.cpu() > 0).any():
            aa, bb, obj_idxs, table_idxs = np.where(predictions.cpu() > 0)
            assert (aa==0).all() and (bb==0).all(), f"{aa} and {bb}"
            for obj_i, table_i in zip(obj_idxs, table_idxs):
                suggestions.append(f"[{predictions.cpu()[0][0][obj_i, table_i]:0.2f}] {self.objects_masterlist[table_i]}: Fetch {self.objects_masterlist[obj_i]}")
        if (predictions.cpu() < 0).any():
            aa, bb, obj_idxs, table_idxs = np.where(predictions.cpu() < 0)
            assert (aa==0).all() and (bb==0).all(), f"{aa} and {bb}"
            for obj_i, table_i in zip(obj_idxs, table_idxs):
                suggestions.append(f"[{-predictions.cpu()[0][0][obj_i, table_i]:0.2f}] {self.objects_masterlist[table_i]}: Return {self.objects_masterlist[obj_i]}")
        
        suggestions.sort(key=lambda x: float(x.split("]")[0].replace("[", "")), reverse=True)
        return suggestions
            
    def run_game(self):
        """Main game loop."""
        VERBOSE and print("🎮 Welcome to the Interactive RoboCraft Game!")
        VERBOSE and print("This game simulates object movements across tables in a crafting environment.")
        VERBOSE and print("You can input actions like 'tableA: Fetch paint' or 'tableB: Return glue'")
        VERBOSE and print("The AI model will provide suggestions based on the current state.\n")
        
        while self.current_time <= self.max_time:
            # Display current state
            self.display_current_state()
            
            # Get AI suggestions
            suggestions = self.get_suggestions()
            if suggestions:
                VERBOSE and print(f"\n🤖 AI Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
            
            # Get user input
            VERBOSE and print(f"\n⏰ Time: {self.minutes_to_time(self.current_time)}")
            print("Enter your action (or 'help' for instructions, 'quit' to exit, '' to move to next time step):")
            
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                VERBOSE and print("👋 Thanks for playing!")
                break
            elif user_input.lower() == 'help':
                self.show_help()
                continue
            elif user_input.lower() == 'skip':
                VERBOSE and print("⏭️  Skipping to next time step...")
                continue
            elif user_input.lower() == '':
                VERBOSE and print("⏭️  No input, moving to next time step...")
                self.current_time += self.dt
                continue
            
            # Parse and validate user input
            timestamp, table, action, object_name = self.decode_movement(f"[{self.minutes_to_time(self.current_time)}] {user_input}")            
            input_time = self.time_to_minutes(timestamp)
            
            # Update game state
            if self.update_state(timestamp, table, action, object_name):
                if self.current_time > self.max_time:
                    VERBOSE and print(f"\n🏁 Game completed! Final time: {self.minutes_to_time(self.max_time)}")
                    break
            else:
                (VERBOSE and print("❌ Action failed. Please try again.")) or print("❌ Action failed. Please try again.")
        
        # Game summary
        self.show_game_summary()
    
    def show_help(self):
        """Display help information."""
        VERBOSE and print("\n📖 Game Help:")
        print("  Action Format: [HH:MM] tableX: Action object_name")
        print("  Examples:")
        print("    tableA: Fetch paint")
        print("    tableB: Return glue")
        print("    tableC: Fetch coasters")
        print("\n  Commands:")
        print("    help  - Show this help")
        print("    skip  - Skip to next time step")
        print("    quit  - Exit the game")
        print("\n  Valid Tables: tableA, tableB, tableC, tableD, tableE")
        print("  Valid Actions: Fetch, Return")
        print("  Valid Objects: " + ", ".join(self.objects_masterlist))
    
    def show_game_summary(self):
        """Display game summary at the end."""
        VERBOSE and print(f"\n{'='*60}")
        VERBOSE and print("🎯 Game Summary")
        VERBOSE and print(f"{'='*60}")
        print(f"Total Actions: {len(self.action_history)}")
        print(f"Final Time: {self.minutes_to_time(self.current_time)}")
        print(f"Total Timesteps: {len(self.scene_graph_history)}")
        
        if self.action_history:
            print(f"\nAction Timeline:")
            for action in self.action_history:
                print(f"  [{action['timestamp']}] {action['table']}: {action['action']} {action['object']}")


def main():
    """Main function to run the interactive game."""
    parser = argparse.ArgumentParser(description='Interactive RoboCraft Game')
    parser.add_argument('--weights', type=str, 
        default="/coc/flash5/mpatel377/repos/SLaTe-PRO/logs_0816_emb_concat_variations_gpt_valid_all/gpt_valid_all/default_100/weights.pt", 
        help='Path to model weights file (.pt)')
    
    args = parser.parse_args()
    
    # Validate weights file
    if not os.path.exists(args.weights):
        VERBOSE and print(f"❌ Error: Weights file not found: {args.weights}")
        return
    
    config = os.path.join(os.path.dirname(args.weights), 'config.json')
    
    try:
        # Create and run the game
        game = InteractiveGame(args.weights, config)
        game.run_game()
        
    except KeyboardInterrupt:
        VERBOSE and print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        VERBOSE and print(f"\n❌ Error running game: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
