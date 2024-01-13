# Use the official Ubuntu image as the base image
FROM ubuntu:18.04

# Set the working directory
WORKDIR /app

# Copy your source code, Makefile, or other necessary files into the container
COPY ./src /src/

# Moves to the /src/ directory on the container file system
WORKDIR /src/

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
        gcc \
        g++ \
        make \
        openmpi-bin \
        libopenmpi-dev \
        libomp-dev \
        ffmpeg \
        mpv \
        nano \
        openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && make

# Set the entry point to bash
CMD /bin/bash