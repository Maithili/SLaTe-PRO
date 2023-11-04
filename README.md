# Sequential Latent Temporal model for Predicting Routine Object Usage (SLaTe-PRO)

SLaTe-PRO, presented in paper [Predicting Routine Object Usage for Proactive Robot Assistance](https://openreview.net/forum?id=rvh0vkwKUM) learns a shared latent space across observation domains to represent user's routine behavior and perform predictions. It is composed of autoencoders to encode each observation domain into the latent space and a recurrent model to perform predictions in this space.

This repository includes 
- SLaTe-PRO model and training code
- [HOMER+ dataset](https://github.com/Maithili/HOMER_PLUS), which is based on the [HOMER dataset](https://github.com/GT-RAIL/rail_tasksim/tree/homer/routines)
- Checkpoints of SLaTe-PRO trained on the above HOMER+ dataset


## How-Tos

To run a previously trained model, run the below code for one of HOuseholdA, HouseholdB or HouseholdC

>`python ./run.py --activity_availability=100 --path=./data/HOMER+/householdA --logs_dir=./logs --ckpt_dir=./checkpoints/HouseholdA/default_100 --read_ckpt`

To train a model, run the below code for one of HOuseholdA, HouseholdB or HouseholdC

>`python ./run.py --activity_availability=100 --path=./data/HOMER+/householdA --logs_dir=./logs`





## Citation
If this work proved helpful, consider citing it as:
```
@inproceedings{patel2023predicting,
  title={Predicting Routine Object Usage for Proactive Robot Assistance},
  author={Patel, Maithili and Prakash, Aswin and Chernova, Sonia},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
