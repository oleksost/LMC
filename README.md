## This repository contains the codebase for the paper **Continual Learning via Local Module Composition**.
---
## Setting up the environemnt
Create a new conda environment and install the requirements.
```
conda create --name ENV python=3.7
conda activate ENV
pip install -r requirements.txt
pip install -e Utils/ctrl/
pip install Utils/nngeometry/
```
---
## CTrL Benchmark
All experiments were run on Nvidia Quadro RTX 8000 GPUs. To run CTrL experiments use the following comands for different streams:

**Stream S<sup>-</sup>**

LMC (task agnostic)
```
python main_transfer.py --activate_after_str_oh=0 --momentum_bn 0.1 --track_running_stats_bn 1 --pr_name lmc_cr --shuffle_test 0 --init_oh=none --task_sequence s_minus --momentum_bn_decoder=0.1 --activation_structural=sigmoid --deviation_threshold=4 --depth=4 --epochs=100 --fix_layers_below_on_addition=0 --hidden_size=64 --lr=0.001 --mask_str_loss=1 --module_init=mean --multihead=gated_linear --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=20 --reg_factor=10  --running_stats_steps=100 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --temp=0.1 --wdecay=0.001
```
(test acc. 0.6863, 15 modules)

MNTDP (task aware)
```
python main_transfer_mntdp.py --momentum_bn 0.1 --pr_name lmc_cr --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_minus --gating MNTDP --shuffle_test 0 --epochs 100 --lr 1e-3 --wdecay 1e-3
```
(test acc. 0.667, 12 modules)

**Stream S<sup>+</sup>**

LMC
```
python main_transfer.py --activate_after_str_oh=0 --activation_structural=sigmoid --deviation_threshold=1.5 --early_stop_complete=0 --pr_name lmc_cr --epochs=100 --epochs_str_only_after_addition=1 --hidden_size=64 --init_oh=none --init_runingstats_on_addition=1 --keep_bn_in_eval_after_freeze=1 --lr=0.001 --module_init=most_likely --momentum_bn=0.1 --momentum_bn_decoder=0.1 --multihead=gated_linear --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=5 --reg_factor=10 --running_stats_steps=100 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --task_sequence=s_plus --temp=1 --wdecay=0.001
```
(test acc. 0.6244, 22 modules)

MNTDP (task aware)
```
python main_transfer_mntdp.py --momentum_bn 0.1 --pr_name lmc_cr --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_plus --gating MNTDP --shuffle_test 0 --epochs 100 --lr 1e-3 --wdecay 1e-3 --regenerate_seed 0
```
(test acc. 0.609, 18 modules)

**Stream S<sup>in</sup>**

LMC
```
python main_transfer.py --activate_after_str_oh=0 --momentum_bn 0.1 --track_running_stats_bn 1 --pr_name lmc_cr --shuffle_test 0 --init_oh=none --task_sequence s_in --momentum_bn_decoder=0.1 --activation_structural=sigmoid --deviation_threshold=4 --depth=4 --epochs=100 --fix_layers_below_on_addition=0 --hidden_size=64 --lr=0.001 --mask_str_loss=1 --module_init=most_likely --multihead=gated_linear --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=20 --reg_factor=10  --running_stats_steps=100 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --temp=0.1 --wdecay=0.001
```
(test acc. 0.7081, 21 modules)

MNTDP (task aware)
```
python main_transfer_mntdp.py --momentum_bn 0.1 --pr_name lmc_cr --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_in --gating MNTDP --shuffle_test 0 --epochs 100 --lr 1e-3 --wdecay 1e-3 --regenerate_seed 0
```
(test acc. 0.6646, 15 modules)

**Stream S<sup>out</sup>**

LMC
```
python main_transfer.py --activate_after_str_oh=0 --momentum_bn 0.1 --track_running_stats_bn 1 --pr_name lmc_cr --shuffle_test 0 --init_oh=none --task_sequence s_out --momentum_bn_decoder=0.1 --activation_structural=sigmoid --deviation_threshold=4 --depth=4 --epochs=100 --fix_layers_below_on_addition=0 --hidden_size=64 --lr=0.001 --mask_str_loss=1 --module_init=mean --multihead=gated_linear --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=20 --reg_factor=10  --running_stats_steps=100 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --temp=0.1 --wdecay=0.001

```
(test acc. 0.5849, 15 modules)

MNTDP (task aware)
```
python main_transfer_mntdp.py --momentum_bn 0.1 --pr_name lmc_cr --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_out --gating MNTDP --shuffle_test 0 --epochs 100 --lr 1e-3 --wdecay 0 --regenerate_seed 0
```
(test acc. 0.6567, 11 modules)

**Stream S<sup>pl</sup>**

LMC
```
python main_transfer.py --activate_after_str_oh=0 --activation_structural=sigmoid --pr_name lmc_cr --deviation_threshold=1.5 --early_stop_complete=0 --epochs=100 --hidden_size=64 --init_oh=none --init_runingstats_on_addition=0 --keep_bn_in_eval_after_freeze=1 --lr=0.001 --module_init=most_likely --momentum_bn=0.1 --momentum_bn_decoder=0.1 --multihead=gated_linear --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=10 --reg_factor=10 --running_stats_steps=100 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --task_sequence=s_pl --temp=1 --regenerate_seed 0 --wdecay=0.001
```
(test acc. 0.6241, 19 modules)

MNTDP (task aware)
```
python main_transfer_mntdp.py --momentum_bn 0.1 --pr_name lmc_cr --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_pl --gating MNTDP --shuffle_test 0 --epochs 100 --lr 1e-3 --wdecay 1e-4 --regenerate_seed 0
```
(test acc. 0.6391, 18 modules)

---

**Stream S<sup>long30</sup>** -- 30 tasks

LMC (task aware)
```
python main_transfer.py --activate_after_str_oh=0 --activation_structural=sigmoid --deviation_threshold=1.5 --epochs=50 --hidden_size=64 --init_oh=none --keep_bn_in_eval_after_freeze=1 --lr=0.001 --module_init=most_likely --momentum_bn_decoder=0.1 --multihead=gated_linear --n_tasks=100 --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=5 --reg_factor=1 --running_stats_steps=50 --seed=180 --str_prior_factor=1 --str_prior_temp=0.01 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=0 --task_sequence=s_long30 --temp=1 --wdecay=0.001

(test acc. 62.44, 50 modules)
```
MNTDP (task aware)
```
python main_transfer_mntdp.py --epochs=50 --hidden_size=64 --lr=0.001 --module_init=most_likely --multihead=gated_linear --n_tasks=100 --seed=180 --task_sequence=s_long30 --wdecay=0.001
```
(test acc. 64.58, 64 modules)

---

**Stream S<sup>long</sup>** -- 100 tasks

LMC (task aware)
```
python main_transfer.py --activate_after_str_oh=0 --activation_structural=sigmoid --deviation_threshold=4 --epochs=100 --hidden_size=64 --init_oh=none --keep_bn_in_eval_after_freeze=1 --lr=0.001 --module_init=most_likely --momentum_bn_decoder=0.1 --multihead=gated_linear --n_tasks=100 --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=5 --reg_factor=1 --running_stats_steps=50 --seed=180 --str_prior_factor=1 --str_prior_temp=0.01 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=0 --task_sequence=s_long --temp=1 --pr_name s_long_cr --wdecay=0
```
(test acc. 63.88, 32 modules)

MNTDP (task aware)
```
python main_transfer_mntdp.py --momentum_bn 0.1 --n_tasks 100 --hidden_size 64 --searchspace topdown --keep_bn_in_eval_after_freeze 1 --pr_name s_long_cr --copy_batchstats 1 --track_running_stats_bn 1 --wand_notes correct_MNTDP --task_sequence s_long --gating MNTDP --shuffle_test 0 --epochs 50 --lr 1e-3 --wdecay 1e-3
```
(test acc. 68.92, 142 modules)

---
## OOD generalization experiments

LMC
```
python main_transfer.py --regenerate_seed 0 --deviation_threshold=8 --epochs=50 --pr_name lmc_cr --hidden_size=64 --keep_bn_in_eval_after_freeze=0 --lr=0.001 --module_init=none --momentum_bn_decoder=0.1 --normalize_data=1 --optmize_structure_only_free_modules=0 --projection_phase_length=10 --no_projection_phase 0 --reg_factor=10 --running_stats_steps=1000 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=linear_no_act --task_sequence=s_ood --temp=1 --wdecay=0 --task_agnostic_test=0
```

EWC
```
python main_transfer.py --epochs=50 --ewc=1000 --hidden_size=256 --keep_bn_in_eval_after_freeze=0 --lr=0.001 --module_init=none --pr_name lmc_cr --multihead=usual --normalize_data=1  --task_sequence=s_ood --use_structural=0 --wdecay=0 --projection_phase_length=0
```

MNTDP
```
python main_transfer_mntdp.py --epochs=50 --regenerate_seed 0 --hidden_size=64 --keep_bn_in_eval_after_freeze=0 --pr_name lmc_cr --lr=0.01 --module_init=none --multihead=usual --normalize_data=1 --task_sequence=s_ood --use_structural=0 --wdecay=0
```

LMC (no projetion)
```
python main_transfer.py --regenerate_seed 0 --deviation_threshold=8 --epochs=50 --pr_name lmc_cr --hidden_size=64 --keep_bn_in_eval_after_freeze=0 --lr=0.001 --module_init=none --momentum_bn_decoder=0.1 --normalize_data=1 --optmize_structure_only_free_modules=0 --projection_phase_length=0 --no_projection_phase 1 --reg_factor=10 --running_stats_steps=1000 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=linear_no_act --task_sequence=s_ood --temp=1 --wdecay=0
```

---
## Plug and play (combining independently trained modular learners)

```
python main_plug_and_play.py --activate_after_str_oh=0 --activation_structural=sigmoid --deviation_threshold=1.5 --early_stop_complete=0 --epochs=100 --epochs_str_only_after_addition=1 --pr_name lmc_cr --hidden_size=64 --init_oh=none --init_runingstats_on_addition=1 --keep_bn_in_eval_after_freeze=1 --lr=0.001 --module_init=mean --momentum_bn=0.1 --momentum_bn_decoder=0.1 --multihead=gated_linear --n_tasks=3 --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=5 --reg_factor=10 --running_stats_steps=10 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --task_sequence=s_pnp_comp --temp=1 --wdecay=0.001
```
---
A list of hyperparameters used for other baselines can be found in the *baselines.txt* file.

---
## References
* [CTrL Benchmark](https://github.com/facebookresearch/CTrLBenchmark)
* [NNGeometry](https://github.com/tfjgeorge/nngeometry)