turtlesim_goto.py: The demo script used during the tutorial. You can run it
anywhere in our Docker container.

Dockerfile: The file used to define our Docker image. To start the Docker
container, follow the instructions in lab1pre pdf document. When building the
image, the other two files need to be in the build context directory:

- entrypoint.bash

- start-novnc.bash
