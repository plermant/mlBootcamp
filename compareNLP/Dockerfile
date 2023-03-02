FROM python:3.9

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 --trusted-host pypi.org --trusted-host files.pythonhosted.org install -r requirements.txt

COPY . .

CMD streamlit run ./streamlit/streamlitApp.py
