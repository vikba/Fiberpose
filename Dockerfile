# One image for a VPS and for a Hugging Face Docker Space (which runs as uid 1000).
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
WORKDIR /app
ENV PATH=/home/user/.local/bin:$PATH \
    FIBERPOSE_MODELS_DIR=/app/checkpoints \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
COPY --chown=user . .
RUN pip install --no-cache-dir ".[web]"
# Analyse the example once so the numba kernels are compiled and cached in the image.
RUN python -c "import imageio.v3 as iio, fiberpose; from fiberpose.paths import EXAMPLE_IMAGE_PATH as p; fiberpose.analyze(iio.imread(p))"
EXPOSE 7860
CMD ["fiberpose", "--web"]
