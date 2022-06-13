# FROM ubuntu:18.04
FROM python:3.7

ENV PYTHONUNBUFFERED True

# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN useradd -m beeerlian
# RUN chown -R beeerlian:beeerlian /home/beeerlian/
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
# USER beeerlian
# RUN pip install --upgrade pip

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./
# COPY exporter model.h5
# Install production dependencies.
RUN pip install -r requirements.txt



# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 main:app

# FROM ubuntu:18.04
# FROM python:3.7

# # RUN apt-get update && apt-get install -y python3.7 python3.7-pip sudo
# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN useradd -m beeerlian
# RUN chown -R beeerlian:beeerlian /home/beeerlian/



# COPY --chown=beeerlian . /home/beeerlian/app/

# USER beeerlian
# ENV FLASK_APP=/home/beeerlian/app/main.py

# RUN pip install --upgrade pip
# RUN cd /home/beeerlian/app/ && pip install -r requirements.txt

# WORKDIR /home/beeerlian/app

# # ENV PORT=8080
# # ENV HOST=0.0.0.0

# # EXPOSE 8080

# # CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
# ENTRYPOINT python -m gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
# # ENTRYPOINT python -m flask run --host=0.0.0.0



# # FROM python:3.7.9-slim

# # # Install production dependencies.
# # ADD requirements.txt .
# # RUN pip install -r requirements.txt
# # # RUN python app.py

# # # Copy local code to the container image.
# # WORKDIR /app
# # COPY . .

# # # Service must listen to $PORT environment variable.
# # # This default value facilitates local development.
# # ENV PORT 8080


# # # Run the web service on container startup. Here we use the gunicorn
# # # webserver, with one worker process and 8 threads.
# # # For environments with multiple CPU cores, increase the number of workers
# # # to be equal to the cores available.
# # CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:app