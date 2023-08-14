FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the model to the container
COPY xgb_model.joblib .

# Copy the FastAPI app code to the container
COPY main.py .

# Expose the port the FastAPI app will run on
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

docker build -t my-fastapi-app .

docker run -p 8000:8000 my-fastapi-app