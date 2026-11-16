FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PAWNSTAR_MODEL=/model/medium.onnx PAWNSTAR_THREADS=2
WORKDIR /app/bot
COPY lichess/bot/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt onnxruntime==1.32.0 numpy==2.5.4
COPY lichess/bot/ /app/bot/
COPY preprocess/ /app/pawnstar/preprocess/
COPY engine/ /app/pawnstar/engine/
COPY lichess/challenge /usr/local/bin/challenge
RUN chmod +x /usr/local/bin/challenge /app/pawnstar/engine/pawnstar_uci.py
CMD ["python", "lichess-bot.py", "--config", "/app/bot/config.yml", "--disable_auto_logging"]
