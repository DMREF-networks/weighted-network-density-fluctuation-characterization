# wndfc -- reproducibility container (the reference environment).
#   docker build -t wndfc .
#   docker run --rm wndfc                       # runs the invariant test suite
#   docker run --rm -v "$PWD/out:/out" wndfc wndfc --poisson 1600 -L 40 --ntype D --weight length_power --theta 1 --kmax 30 --out /out/run.npz
FROM python:3.10-slim

RUN apt-get update  && apt-get install -y --no-install-recommends build-essential  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/wndfc
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY wndfc ./wndfc
COPY tests ./tests
COPY figures ./figures
RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "pytest", "tests", "-q"]
