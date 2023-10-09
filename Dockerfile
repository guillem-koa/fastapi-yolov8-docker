FROM tiangolo/uvicorn-gunicorn:python3.10

RUN apt update && \
    apt install -y htop libgl1-mesa-glx libglib2.0-0

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80
# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]