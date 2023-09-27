FROM python:3.9

WORKDIR /app

COPY api/requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY api/ ./api

COPY models/model.pk ./models/model.pk

COPY models/column_equivalence.pk ./models/column_equivalence.pk

COPY models/scaler.pk ./models/scaler.pk

COPY initializer.sh .

RUN chmod +x initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]
