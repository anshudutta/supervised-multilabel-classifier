FROM python:3.9.6-buster

RUN apt update && apt upgrade -y
WORKDIR /home/app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
