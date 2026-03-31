FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]