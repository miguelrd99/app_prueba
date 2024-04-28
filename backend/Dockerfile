FROM python:3.11
COPY requirements.txt app_prueba/requirements.txt
WORKDIR /app_prueba
RUN pip install -r requirements.txt
COPY . /app_prueba
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ,"--reload"]