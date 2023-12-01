# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install the required packages directly
RUN pip install --no-cache-dir fastapi==0.68.1 uvicorn==0.15.0 onnxruntime==1.9.0 joblib==1.1.0 scikit-learn==0.24.2  # Include scikit-learn because the deployement crashed witout it

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run your FastAPI application when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

