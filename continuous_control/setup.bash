
# Setting up the enviroment
conda env create -f environment.yml
conda activate continuous_control

# Downloading the unity enviroment

mkdir unity
cd unity

# One agents
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
unzip Reacher_Linux.zip
rm Reacher_Linux.zip
mv Reacher_Linux Reacher_Linux_Single

# Many agents
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip
unzip Reacher_Linux.zip
rm Reacher_Linux.zip

mv Reacher_Linux Reacher_Linux_Many
