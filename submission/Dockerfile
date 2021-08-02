# Get a good base docker image, pin it to a specific SHA digest to ensure reproducibility 
#FROM rocker/r-base@sha256:ec224c21eff00e6cd8016419fae2886596c76e80fb1ae042e657b3cd08ba30d8
FROM nvcr.io/nvidia/pytorch:20.03-py3
# Install dependencies, this example is to use tensorflow in R via the keras library
#RUN add-apt-repository universe
#RUN apt-get update -y 
#RUN apt-get install -y libpython3-dev
#RUN apt-get install -y python3-pip 
#RUN apt-get install libjpeg-dev
RUN pip install virtualenv Pillow pandas
#RUN apt-get install -y python3-venv

RUN pip install piexif
#RUN pip install PIL logging os shutil random
#RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install matplotlib
RUN pip install xmltodict tqdm
#RUN R -e "install.packages(c('tidyverse','keras'), dependencies=TRUE, repos='http://cran.rstudio.com/')"
#RUN R -e 'keras::install_keras(tensorflow = "2.2.0rc2-gpu")'
#RUN echo "Sys.setenv(RETICULATE_PYTHON = '/root/.virtualenvs/r-reticulate/bin/python')" > ~/.Rprofile

# Required: Create /train /test and /output directories 
RUN mkdir /train/
RUN mkdir /test/
RUN mkdir /output/
#RUN mkdir output/logs
#RUN mkdir output/test
#RUN mkdir output/test/narrowing_all
#RUN mkdir output/test/erosion_all

# Required for GPU: run.sh defines PATHs to find GPU drivers, see run.sh for specific commands
COPY run.sh /run.sh
COPY training.csv /usr/local/bin/training.csv
COPY model.py /usr/local/bin/model.py
COPY data.py /usr/local/bin/data.py
COPY preprocessing.py /usr/local/bin/preprocessing.py
COPY 0-Preprocessing.py /usr/local/bin/0-Preprocessing.py
COPY 1-NarrowingDetection.py /usr/local/bin/1-NarrowingDetection.py
COPY 2-ErosionDetection.py /usr/local/bin/2-ErosionDetection.py
COPY 3-NarrowingScore.py /usr/local/bin/3-NarrowingScore.py
COPY 4-ErosionScore.py /usr/local/bin/4-ErosionScore.py
COPY 5-Final.py /usr/local/bin/5-Final.py
COPY pretrained_models /usr/local/bin/pretrained_models
COPY utils /usr/local/bin/utils
COPY models /usr/local/bin/models
#COPY write_test.py /usr/local/bin/write_test.py
# Required: a model file. 
#COPY model.R /usr/local/bin/model.R

# Make model and runfiles executable 
RUN chmod 775 /usr/local/bin/0-Preprocessing.py
RUN chmod 775 /usr/local/bin/1-NarrowingDetection.py
RUN chmod 775 /usr/local/bin/2-ErosionDetection.py
RUN chmod 775 /usr/local/bin/3-NarrowingScore.py
RUN chmod 775 /usr/local/bin/4-ErosionScore.py
RUN chmod 775 /usr/local/bin/5-Final.py
RUN chmod 775 /run.sh
#RUN chmod 775 /usr/local/bin/write_test.py
# This is for the virtualenv defined above, if not using a virtualenv, this is not necessary
RUN chmod 755 /root #to make virtualenv accessible to singularity user

# Required: define an entrypoint. run.sh will run the model for us, but in a different configuration
# you could simply call the model file directly as an entrypoint 
#ENTRYPOINT ["usr/local/bin/nvidia_entrypoint.sh"]
ENTRYPOINT ["/bin/bash", "/run.sh"]
