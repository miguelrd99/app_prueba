FROM python:3.11
WORKDIR /app 
COPY requirements.txt ./requirements.txt
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["sh", "-c", "streamlit run --server.port $PORT /app/streamlit_app.py"]