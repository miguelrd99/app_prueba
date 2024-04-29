FROM python:3.11
COPY . /app_prueba
WORKDIR /app_prueba
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ,"--reload"]