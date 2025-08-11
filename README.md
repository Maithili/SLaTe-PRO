# SLaTe-PRO

## How-Tos

To run a previously trained model

>`python ./run.py --coarse --activity_availability=100 --path=data/HouseholdVariations/householdA --logs_dir=./logs --ckpt_dir=logs/householdA/default_100 --read_ckpt`

To train a model

>`python ./run.py --coarse --activity_availability=100 --path=data/HouseholdVariations/householdA --logs_dir=./logs`


## ToDos

- ~Visualize each day's predictions and GT~
- ~Embeddings for objects~
- Try pretraining on LLM-generated data
- Split results by objects added/removed from table
- Figure out layer and embedding sizes
- Apply query mask for future predictions. Use Case: When someone refuses an object, the robot should use that for the next prediction
- Compare Overhead v.s. Overhead@RobotTimestamps



## Citation

Repository for the model proposed in Patel et al. "Predicting Routine Object Usage for Proactive Robot Assistance", CoRL 2023