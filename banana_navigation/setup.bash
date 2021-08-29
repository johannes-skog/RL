

# Setting up the enviroment
conda env create -f environment.yml
conda activate banana_navigation


# Downloading the unity enviroment
mkdir unity
cd unity
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip
rm Banana_Linux.zip
