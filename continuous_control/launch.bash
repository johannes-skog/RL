mkdir runs

tensorboard --logdir runs/ --port 6006 &

jupyter notebook --NotebookApp.password='' &
