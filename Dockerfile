FROM python:3.7
COPY ./requirements.txt  /docker_app/requirements.txt
WORKDIR /docker_app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /docker_app
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"] 