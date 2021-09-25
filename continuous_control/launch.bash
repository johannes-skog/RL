mkdir runs

conda activate  continuous_control

tensorboard --logdir runs/ --port 6006 &

jupyter notebook --NotebookApp.password='' &
