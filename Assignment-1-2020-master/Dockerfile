FROM python:3.7

 ADD entry_script.py /
 ADD requirements.txt /

 RUN pip install -r requirements.txt
 RUN python -m nltk.downloader stopwords

 ENTRYPOINT [ "python", "./entry_script.py" ]   