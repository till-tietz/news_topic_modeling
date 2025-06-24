# start from Miniconda base image
FROM continuumio/miniconda3

# set working directory
WORKDIR /app

# copy environment definition
COPY environment.yml .

# create conda environment
RUN conda env create -f environment.yml

# activate env by default
SHELL ["conda", "run", "-n", "news_topic_modeling_env", "/bin/bash", "-c"]

# copy src into container
COPY src/ ./

# Set entrypoint to main script
ENTRYPOINT ["conda", "run", "-n", "news_topic_modeling_env", "python", "main.py"]