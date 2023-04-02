FROM alpine:3.17.2

ENV PYTHONUNBUFFERED=1

RUN addgroup -g 1000 python
RUN adduser -u 1000 -G python --disabled-password python

RUN apk add --update --no-cache python3 python3-dev pkgconfig g++ zlib-dev \
    jpeg-dev gfortran cmake openblas openblas-dev freetype freetype-dev

RUN ln -sf python3 /usr/bin/python \
    && python3 -m ensurepip \
    && pip3 install --no-cache --upgrade pip setuptools pycodestyle

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

WORKDIR /work

USER python:python