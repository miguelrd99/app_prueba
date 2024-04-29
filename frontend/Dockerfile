FROM python:3.11
COPY . /app_prueba
WORKDIR /app_prueba
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["streamlit_app.py"]