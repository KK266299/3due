python ue_generate.py \
  dataset=brats19 \
  task.run_name=minmin_noise_step_0.0039215 \
  method=min_min \
  task=brats19_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.step_size=0.0039215 \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.algorithm.params.surrogate_step=2 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10

python ue_generate.py \
  dataset=brats19 \
  task.run_name=minmin_noise_step_1e-2 \
  method=min_min \
  task=brats19_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.step_size=1e-2 \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.algorithm.params.surrogate_step=2 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10

python ue_generate.py \
  dataset=brats19 \
  task.run_name=minmin_noise_step_1e-3 \
  method=min_min \
  task=brats19_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.step_size=1e-3 \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.algorithm.params.surrogate_step=2 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10

  python ue_generate.py \
  dataset=brats19 \
  task.run_name=minmin_noise_step_5e-3 \
  method=min_min \
  task=brats19_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.step_size=5e-3 \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.algorithm.params.surrogate_step=10 \
  ue.io.save_from_epoch=0 \
  ue.io.save_every=10