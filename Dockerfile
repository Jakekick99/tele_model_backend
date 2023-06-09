# Base image
FROM python:3.9-alpine

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory
WORKDIR /myapp

# Install dependencies
COPY requirements.txt .
RUN pip install docker
RUN pip install pillow
RUN pip install numpy
RUN pip install tensorflow
RUN pip install scikit-learn
RUN pip install hugginface-hub

# Copy the Django project
COPY . .

# Expose the desired port
EXPOSE 8094

# Run the Django server on localhost and port 8094
CMD ["python", "manage.py", "runserver", "localhost:8094"]