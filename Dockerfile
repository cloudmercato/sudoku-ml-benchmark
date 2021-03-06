FROM tensorflow/tensorflow 

WORKDIR /sudoku-ml-bench
ADD . /sudoku-ml-bench

RUN mkdir -p /log_dir/
RUN mkdir -p /models/

RUN pip install https://github.com/cloudmercato/sudoku-game/archive/refs/heads/master.zip
RUN pip install https://github.com/cloudmercato/sudoku-ml/archive/refs/heads/master.zip
RUN python setup.py develop

VOLUME /log_dir/
VOLUME /models/

CMD ["sudoku-ml-bench", "--train-dataset-size", "100000", "--infer-dataset-size", "100"]

EXPOSE 6006/TCP
EXPOSE 6007/TCP
