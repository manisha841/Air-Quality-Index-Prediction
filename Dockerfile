FROM python:3.8

WORKDIR /app

RUN apt update -y && \
    apt install -y python3-pip 


COPY requirements.txt ./

RUN pip3 --no-cache-dir install -r requirements.txt

COPY . .
EXPOSE 8501

CMD streamlit run main.py