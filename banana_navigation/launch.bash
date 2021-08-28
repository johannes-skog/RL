mkdir runs

tensorboard --logdir runs/ --port 6006 & disown


jupyter notebook --NotebookApp.password='' & 
