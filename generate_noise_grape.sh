python ue_generate.py \
  dataset=brats19 \
  task.run_name=minmin_noise \
  method=min_min \
  task=brats19_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.algorithm.params.step_size=0.0007843 \
  training.batch_size=8\
  training.gpu_ids=[0] \
  ue.algorithm.params.surrogate_step=10 \
  ue.io.save_from_epoch=50 \
  ue.io.save_every=10