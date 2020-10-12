FROM python:3.6.9
WORKDIR /Project
ADD . /Project
RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN pip3 install -r requirements.txt
RUN export FLASKAPP=./app.py
EXPOSE 5000
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]
