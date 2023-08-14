FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the model to the container
COPY xgb.model.joblib .

# Copy the FastAPI app code to the container
COPY app.main .

# Expose the port the FastAPI app will run on
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]